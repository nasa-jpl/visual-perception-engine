import os
import socket
import time
import multiprocessing as mp
from multiprocessing import reduction
from math import prod
from typing import Literal

import torch
from cuda import cuda, cudart, nvrtc


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class CUDASharedMemorySlot:
    """This class is a container for a set of cuda tensors characterized by the data signature.
    WARNING: this class is not thread-safe."""

    def __init__(
        self,
        data_signature: dict[str, tuple[int]],
        dtype: torch.dtype,
        is_original_process,
        device_id: int = 0,
        add_batch_dim: bool = False,
    ):
        # store necessary parameters
        self.dtype = dtype
        self.is_original_process = is_original_process
        self.data_signature = data_signature
        self.device_id = device_id

        # get additional parameters
        self.logger = mp.get_logger()
        batch = (1,) if add_batch_dim else ()
        self.effective_data_signature = {key: (*batch, *value) for key, value in data_signature.items()}
        self.original_nbytes = {
            key: prod(value) * dtype.itemsize for key, value in self.effective_data_signature.items()
        }
        self._shareable_handles = None  # only for the original process

        # Make sure that the system meets the requirements such as virtual address management
        if not self.check_system_properties():
            raise RuntimeError("System does not meet the requirements for using CUDASharedMemorySlot")

        # Get minimum granularity
        status, min_granularity = cuda.cuMemGetAllocationGranularity(
            self.get_allocation_properties(), cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.logger.error(
                "cuMemGetAllocationGranularity returned error code {}: {}".format(status, cuda.cuGetErrorName(status))
            )
            raise RuntimeError("Failed to get minimum granularity")
        self.min_granularity = min_granularity
        self.allocated_nbytes = {
            key: self.round_up(value, self.min_granularity) for key, value in self.original_nbytes.items()
        }

        # dict containing pointers to memory
        self._memory = {}

        # if this is the original process, allocate memory and generate shareable handles
        if is_original_process:
            self._shareable_handles = {}
            for key, value in self.effective_data_signature.items():
                status, dptr, shareableHandle, allocated_nbytes = self.allocate_shareable_memory(
                    self.original_nbytes[key]
                )
                if status != cuda.CUresult.CUDA_SUCCESS:
                    self.close()
                    self.logger.error("Failed to allocate shareable memory")
                    raise RuntimeError("Failed to allocate shareable memory")

                self._memory[key] = dptr
                assert self.allocated_nbytes[key] == allocated_nbytes, (
                    "Empirical allocation size does not match expected value"
                )
                self.shareable_handles[key] = shareableHandle
                os.set_inheritable(shareableHandle, True)  # make the handle inheritable by child processes

    def __getstate__(self):
        """Return the state of the object for pickling. Intended to be used only when passing the object to a child process.
        The new state will not contain the memory pointers, which will need to be recovered by calling receive_shareable_handles.
        Furhtermore, the new state will contain a flag indicating that this is not the original process.
        """
        state = self.__dict__.copy()
        state["_memory"] = {}
        state["is_original_process"] = False
        state["_shareable_handles"] = None
        return state

    def __setstate__(self, state):
        """Set the state of the object after unpickling. Intended to be used only when passing the object to a child process.
        The new state will contain a flag indicating that this is not the original process.
        """
        self.__dict__.update(state)
        self.is_original_process = False
        self._shareable_handles = None
        self._memory = {}

    @property
    def shareable_handles(self):
        if self.is_original_process:
            return self._shareable_handles
        else:
            raise RuntimeError("Shareable handles are only available in the original process")

    @property
    def device(self):
        return checkCudaErrors(cuda.cuDeviceGet(self.device_id))

    def send_shareable_handles(self, parent_socket, child_pid):
        for value in self.shareable_handles.values():
            reduction.send_handle(parent_socket, value, child_pid)

    def receive_shareable_handles(self, child_socket):
        for key in self.data_signature.keys():
            # NOTE: this introduces blocking behavior
            shareable_handle = reduction.recv_handle(child_socket)

            status, dptr = self.ptr_from_shareable_handle(shareable_handle, self.allocated_nbytes[key])
            if status != cuda.CUresult.CUDA_SUCCESS:
                self.close()
                self.logger.error("Failed to import shareable handle")
                raise RuntimeError("Failed to import shareable handle")

            self._memory[key] = dptr

    def write(self, source: dict[str, torch.Tensor], sync: bool = True):
        stream_identifier = torch.cuda.current_stream().cuda_stream  # use the same stream for torch and cuda operations
        cpy = cuda.cuMemcpyHtoDAsync if list(source.values())[0].device.type == "cpu" else cuda.cuMemcpyDtoDAsync
        for key, value in source.items():
            assert value.shape == self.effective_data_signature[key], (
                f"Shape mismatch for key {key}. Expected {self.effective_data_signature[key]}, got {value.shape}"
            )
            checkCudaErrors(cpy(self._memory[key], value.data_ptr(), self.original_nbytes[key], stream_identifier))

        if sync:
            # Synchronize the stream, 0 indicated default stream
            checkCudaErrors(cuda.cuStreamSynchronize(stream_identifier))

    def read(self, destination: dict[str, torch.Tensor], sync: bool = True):
        """Read the data from the shared memory to the destination tensors."""
        stream_identifier = torch.cuda.current_stream().cuda_stream  # use the same stream for torch and cuda operations
        cpy = cuda.cuMemcpyDtoHAsync if list(destination.values())[0].device.type == "cpu" else cuda.cuMemcpyDtoDAsync
        for key, shmem_ptr in self._memory.items():
            checkCudaErrors(cpy(destination[key].data_ptr(), shmem_ptr, self.original_nbytes[key], stream_identifier))
            assert destination[key].shape == self.effective_data_signature[key], (
                f"Shape mismatch for key {key}. Expected {self.effective_data_signature[key]}, got {destination[key].shape}"
            )

        if sync:
            # Synchronize the stream
            checkCudaErrors(cuda.cuStreamSynchronize(stream_identifier))

    def get_non_shared_empty_memory_slot(self, device: Literal["cpu", "cuda"]) -> dict[str, torch.Tensor]:
        """Return a dictionary containing empty tensors with the same shape as the data signature."""
        data = {}
        for key, value in self.effective_data_signature.items():
            data[key] = torch.empty(size=value, dtype=self.dtype, device=device)
        return data

    def close(self):
        for key, value in self._memory.items():
            self.cleanup_memory(value, self.allocated_nbytes[key])

    def check_system_properties(self) -> bool:
        # Check that the selected device supports virtual address management
        attributeVal = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, self.device
            )
        )
        if not attributeVal:
            self.logger.error("Device {} doesn't support VIRTUAL ADDRESS MANAGEMENT.".format(self.device))
            return False

        attributeVal = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, self.device
            )
        )
        if not attributeVal:
            self.logger.error("Device {} doesn't support  POSIX FILE DESCRIPTORS.".format(self.device))
            return False

        return True

    @staticmethod
    def round_up(x, y):
        return int((x - 1) / y + 1) * y

    def ptr_from_shareable_handle(self, shareableHandle, nbytes):
        # Import pointer from shareable handle
        prop = self.get_allocation_properties()
        status, imported_allocation_handle = cuda.cuMemImportFromShareableHandle(
            shareableHandle, prop.requestedHandleTypes
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.logger.error(
                "cuMemImportFromShareableHandle returned error code {}: {}".format(status, cuda.cuGetErrorName(status))
            )
            return status, None

        # Set up memory
        status, dptr = self.setup_memory_allocation(nbytes, imported_allocation_handle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None

        # Release the imported allocation handle
        (status,) = cuda.cuMemRelease(imported_allocation_handle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, nbytes)
            self.logger.error("cuMemRelease returned error code {}: {}".format(status, cuda.cuGetErrorName(status)))
            return status, None

        return status, dptr

    def cleanup_memory(self, dptr, size):
        # Unmap the mapped virtual memory region
        # Since the handles to the mapped backing stores have already been released
        # by cuMemRelease, and these are the only/last mappings referencing them,
        # The backing stores will be freed.
        # Since the memory has been unmapped after this call, accessing the specified
        # va range will result in a fault (unitl it is remapped).
        # Note that the physical address is demapped immediately after unmapping
        status = cuda.cuMemUnmap(dptr, size)
        if status[0] != cuda.CUresult.CUDA_SUCCESS:
            return status

        # Free the virtual address region.  This allows the virtual address region
        # to be reused by future cuMemAddressReserve calls.  This also allows the
        # virtual address region to be used by other allocation made through
        # opperating system calls like malloc & mmap.
        status = cuda.cuMemAddressFree(dptr, size)
        if status[0] != cuda.CUresult.CUDA_SUCCESS:
            return status
        return status

    def get_allocation_properties(self) -> cuda.CUmemAllocationProp:
        """Specify allocation properties"""
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        return prop

    @staticmethod
    def get_access_descriptor(prop) -> cuda.CUmemAccessDesc:
        accessDescriptor = cuda.CUmemAccessDesc()
        accessDescriptor.location.type = prop.location.type
        accessDescriptor.location.id = prop.location.id
        accessDescriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        return accessDescriptor

    def setup_memory_allocation(
        self, allocated_nbytes: int, allocation_handle: cuda.CUmemGenericAllocationHandle
    ) -> tuple[cuda.CUresult, int]:
        # Reserve virtual address space
        status, dptr = cuda.cuMemAddressReserve(allocated_nbytes, 0, cuda.CUdeviceptr(0), 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, allocated_nbytes)
            self.logger.error(
                "cuMemAddressReserve returned error code {}: {}".format(status, cuda.cuGetErrorName(status))
            )
            return status, None

        # Map physical memory into virtual address space
        (status,) = cuda.cuMemMap(int(dptr), allocated_nbytes, 0, allocation_handle, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, allocated_nbytes)
            self.logger.error("cuMemMap returned error code {}: {}".format(status, cuda.cuGetErrorName(status)))
            return status, None

        # Set access permissions on the mapped region
        prop = self.get_allocation_properties()
        access_descriptors = [self.get_access_descriptor(prop)]
        (status,) = cuda.cuMemSetAccess(dptr, allocated_nbytes, access_descriptors, len(access_descriptors))
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, allocated_nbytes)
            self.logger.error("cuMemSetAccess returned error code {}: {}".format(status, cuda.cuGetErrorName(status)))
            return status, None

        return status, dptr

    def allocate_shareable_memory(self, nbytes) -> tuple[cuda.CUresult, int, int, int]:
        prop = self.get_allocation_properties()

        # Round up requested size to be multiple of minimum granularity
        allocated_nbytes = self.round_up(nbytes, self.min_granularity)

        # Allocate memory
        status, allocationHandle = cuda.cuMemCreate(allocated_nbytes, prop, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            # no need for cleanup, since the allocation failed
            self.logger.error("cuMemCreate returned error code {}: {}".format(status, cuda.cuGetErrorName(status)))
            return status, None, None, None

        status, dptr = self.setup_memory_allocation(allocated_nbytes, allocationHandle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            return status, None, None, None

        # Export allocation handle to shareable handle
        status, shareableHandle = cuda.cuMemExportToShareableHandle(allocationHandle, prop.requestedHandleTypes, 0)
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, allocated_nbytes)
            self.logger.error(
                "cuMemExportToShareableHandle returned error code {}: {}".format(status, cuda.cuGetErrorName(status))
            )
            return status, None, None, None

        # Release allocation handle
        (status,) = cuda.cuMemRelease(allocationHandle)
        if status != cuda.CUresult.CUDA_SUCCESS:
            self.cleanup_memory(dptr, allocated_nbytes)
            self.logger.error("cuMemRelease returned error code {}: {}".format(status, cuda.cuGetErrorName(status)))
            return status, None, None, None

        return status, dptr, shareableHandle, allocated_nbytes


class CUDASharedMemorySlotWithID(CUDASharedMemorySlot):
    """A shared memory slot that stores tensors and an ID. Not thread safe."""

    def __init__(
        self,
        data_signature: dict[str, tuple[int]],
        dtype: torch.dtype,
        is_original_process: bool,
        device_id: int = 0,
        add_batch_dim: bool = False,
    ):
        super().__init__(data_signature, dtype, is_original_process, device_id, add_batch_dim)
        self.id = mp.Value("d", -1, lock=False)

    def write(self, source: dict[str, torch.Tensor], id: int, sync: bool) -> None:
        """Write the data and update the ID."""
        self.id.value = id
        super().write(source, sync)

    def read(self, destination: dict[str, torch.Tensor], sync: bool) -> tuple[int, dict[str, torch.Tensor]]:
        """Read the data and the ID. NOTE destination tensor is modified in place"""
        super().read(destination, sync)
        return self.get_id(), destination

    def get_id(self) -> int:
        """Get the ID."""
        return self.id.value


class CUDASharedMemorySlotWithIDAndTimestamp(CUDASharedMemorySlotWithID):
    """A shared memory slot that stores tensors and a timestamp. Not thread safe."""

    def __init__(
        self,
        data_signature: dict[str, tuple[int]],
        dtype: torch.dtype,
        is_original_process: bool,
        device_id: int = 0,
        add_batch_dim: bool = False,
    ):
        super().__init__(data_signature, dtype, is_original_process, device_id, add_batch_dim)
        self.timestamp = mp.Value("d", float("-inf"), lock=False)

    def write(self, source: dict[str, torch.Tensor], id: int, sync: bool) -> None:
        """Write the data and update the timestamp."""
        self.timestamp.value = time.perf_counter()
        super().write(source, id, sync)

    def read(self, destination: dict[str, torch.Tensor], sync: bool) -> tuple[int, float, dict[str, torch.Tensor]]:
        """Read the data and the timestamp. NOTE destination tensor is modified in place"""
        id, destination = super().read(destination, sync)
        return id, self.get_timestamp(), destination

    def get_timestamp(self) -> float:
        """Get the timestamp."""
        return self.timestamp.value


# consumer
def worker(shared_mem_slot: CUDASharedMemorySlot, socket):
    checkCudaErrors(cuda.cuInit(0))
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, shared_mem_slot.device))
    output_slot = shared_mem_slot.get_non_shared_empty_memory_slot("cuda")
    shared_mem_slot.receive_shareable_handles(socket)
    shared_mem_slot.read(output_slot)
    print(output_slot)
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
    socket.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    dtype = torch.float32

    # producer
    checkCudaErrors(cuda.cuInit(0))
    cu_device = checkCudaErrors(cuda.cuDeviceGet(0))
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cu_device))
    shared_mem_slot = CUDASharedMemorySlot({"a": (2, 2), "b": (1,)}, dtype, True, add_batch_dim=True)

    data = {"a": torch.rand((2, 2), dtype=dtype, device="cuda"), "b": torch.rand((1,), dtype=dtype, device="cuda")}
    print(data)
    shared_mem_slot.write(data)

    parent_sock, child_sock = socket.socketpair()
    os.set_inheritable(parent_sock.fileno(), True)
    os.set_inheritable(child_sock.fileno(), True)

    p = mp.Process(target=worker, args=(shared_mem_slot, child_sock))
    p.start()

    shared_mem_slot.send_shareable_handles(parent_sock, p.pid)
    parent_sock.close()

    p.join()
    shared_mem_slot.close()
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
