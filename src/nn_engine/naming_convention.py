# FM = Foundation Model
# MH = Model Head

PREPROCESSING_INPUT = "preprocessing_input"  # input to the preprocessing module in the foundation model node
FM_INPUT = "fm_input"  # input to the foundation model
FM_INTERMEDIATE_FEATURES_1 = "fm_intermediate_features_1"  # intermediate features of the foundation model
FM_INTERMEDIATE_CLS_TOKEN_1 = "fm_intermediate_cls_token_1"  # intermediate cls token of the foundation model
FM_INTERMEDIATE_FEATURES_2 = "fm_intermediate_features_2"  # intermediate features of the foundation model
FM_INTERMEDIATE_CLS_TOKEN_2 = "fm_intermediate_cls_token_2"  # intermediate cls token of the foundation model
FM_INTERMEDIATE_FEATURES_3 = "fm_intermediate_features_3"  # intermediate features of the foundation model
FM_INTERMEDIATE_CLS_TOKEN_3 = "fm_intermediate_cls_token_3"  # intermediate cls token of the foundation model
FM_OUTPUT_FEATURES = "fm_output_features"  # output features of the foundation model
FM_OUTPUT_CLS_TOKEN = "fm_output_cls_token"  # output cls token of the foundation model
MH_OUTPUT = "mh_output"  # output of the model head
MH_OBJECT_DETECTION_LABELS = "mh_object_detection_labels"  # object detection labels of the model head
MH_OBJECT_DETECTION_BOXES_NORMALIZED = (
    "mh_object_detection_boxes_normalized"  # normalized object detection boxes of the model head
)
MH_OBJECT_DETECTION_SCORES = "mh_object_detection_scores"  # object detection scores of the model head
POSTPROCESSING_OUTPUT = "postprocessing_output"  # output of the postprocessing module in the model head node
