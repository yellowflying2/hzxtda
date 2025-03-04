#!/usr/bin/env python
import os
import argparse
import torch
import logging
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

from wsovod.config import add_wsovod_config
from wsovod.engine import DefaultTrainer_WSOVOD, DefaultTrainer_WSOVOD_MixedDatasets
from wsovod.modeling.test_time_adaptation import (
    update_cache, compute_cache_logits, softmax_entropy, get_entropy
)

logger = logging.getLogger("wsovod")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_wsovod_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Enable TDA if requested
    if args.use_tda:
        cfg.MODEL.ROI_HEADS.USE_TDA = True
        cfg.MODEL.ROI_HEADS.POS_CACHE_ENABLED = True
        cfg.MODEL.ROI_HEADS.NEG_CACHE_ENABLED = True
        
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="wsovod")
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description="WSOVOD Testing with TDA")
    parser.add_argument(
        "--config-file", 
        default="", 
        metavar="FILE", 
        help="path to config file"
    )
    parser.add_argument(
        "--eval-only", 
        action="store_true", 
        help="perform evaluation only"
    )
    parser.add_argument(
        "--use-tda", 
        action="store_true", 
        help="use Test-Time Domain Adaptation"
    )
    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=1, 
        help="number of gpus"
    )
    parser.add_argument(
        "opts", 
        default=None, 
        nargs=argparse.REMAINDER, 
        help="modify config options using the command-line"
    )
    return parser.parse_args()

def main(args):
    cfg = setup(args)
    logger.info("Running with config:\n{}".format(cfg))
    
    # Determine which model to use
    if "MixedDatasets" in args.config_file:
        TrainerClass = DefaultTrainer_WSOVOD_MixedDatasets
    else:
        TrainerClass = DefaultTrainer_WSOVOD
    
    # Create model
    trainer = TrainerClass(cfg)
    model = trainer.build_model(cfg)
    
    # Load checkpoint
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Add TDA functionality to the model if enabled
    if cfg.MODEL.ROI_HEADS.USE_TDA:
        # Apply the TDA patch to WSOVODROIHeads
        monkey_patch_roi_heads_for_tda(model, cfg)
        logger.info("TDA is enabled for testing")
    
    # Run evaluation
    results = {}
    results = trainer.test_WSL(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        results.update(trainer.test_with_TTA_WSL(cfg, model))
    
    return results

def monkey_patch_roi_heads_for_tda(model, cfg):
    """
    Apply Test-Time Domain Adaptation functionality to the model's ROI heads.
    """
    # Get the ROI heads module
    roi_heads = model.roi_heads
    
    # Add TDA attributes
    roi_heads.use_tda = cfg.MODEL.ROI_HEADS.USE_TDA
    roi_heads.pos_cache_config = {
        "enabled": cfg.MODEL.ROI_HEADS.POS_CACHE_ENABLED,
        "shot_capacity": cfg.MODEL.ROI_HEADS.POS_CACHE_CAPACITY,
        "alpha": cfg.MODEL.ROI_HEADS.POS_CACHE_ALPHA,
        "beta": cfg.MODEL.ROI_HEADS.POS_CACHE_BETA,
    }
    roi_heads.neg_cache_config = {
        "enabled": cfg.MODEL.ROI_HEADS.NEG_CACHE_ENABLED,
        "shot_capacity": cfg.MODEL.ROI_HEADS.NEG_CACHE_CAPACITY,
        "alpha": cfg.MODEL.ROI_HEADS.NEG_CACHE_ALPHA,
        "beta": cfg.MODEL.ROI_HEADS.NEG_CACHE_BETA,
        "entropy_threshold": {
            "lower": cfg.MODEL.ROI_HEADS.NEG_ENTROPY_LOWER,
            "upper": cfg.MODEL.ROI_HEADS.NEG_ENTROPY_UPPER,
        },
        "mask_threshold": {
            "lower": cfg.MODEL.ROI_HEADS.NEG_MASK_LOWER,
            "upper": cfg.MODEL.ROI_HEADS.NEG_MASK_UPPER,
        },
    }
    roi_heads.pos_cache = {}
    roi_heads.neg_cache = {}
    
    # Save original _forward_box method
    original_forward_box = roi_heads._forward_box
    
    # Create TDA-enhanced _forward_box method
    def tda_forward_box(
        self, 
        features: dict, 
        proposals: list, 
        data_aware_features=None, 
        classifier=None,
        append_background=True,
        file_names=None,
        loaded_proposals=None,
    ):
        # If in training mode, use the original method
        if self.training:
            return original_forward_box(
                features, proposals, data_aware_features, 
                classifier, append_background, file_names, loaded_proposals
            )
        
        # For inference, apply TDA
        features_list = [features[f] for f in self.box_in_features]
        if self.mrrp_on:
            features_list = [torch.chunk(f, self.mrrp_num_branch) for f in features_list]
            features_list = [ff for f in features_list for ff in f]

        box_features = self.box_pooler(
            features_list,
            [x.proposal_boxes for x in proposals],
            level_ids=[torch.div(x.level_ids, 1000, rounding_mode='floor') for x in proposals] if self.mrrp_on else None,
        )
        
        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        if self.pooler_type == "ROILoopPool":
            objectness_logits = torch.cat(
                [objectness_logits, objectness_logits, objectness_logits], dim=0
            )
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        box_features = self.box_head(box_features)
        
        if self.pooler_type == "ROILoopPool":
            box_features, box_features_frame, box_features_context = torch.chunk(
                box_features, 3, dim=0
            )
            if data_aware_features is not None:
                box_features = box_features + data_aware_features
                box_features_frame = box_features_frame + data_aware_features
                box_features_context = box_features_context + data_aware_features
            del box_features_frame
            del box_features_context
        else:
            if data_aware_features is not None:
                box_features += data_aware_features
        
        if self.refine_K > 0:
            # Get initial predictions
            predictions_K = []
            for k in range(self.refine_K):
                predictions_k = self.box_refinery[k](box_features, classifier, append_background)
                predictions_K.append(predictions_k)
            
            # Apply TDA if enabled
            if self.use_tda:
                logger.info("Applying TDA to refine detections")
                final_scores, final_deltas = predictions_K[-1]
                
                # Get confidences and predicted classes
                confidences = torch.softmax(final_scores, dim=1)
                pred_classes = torch.argmax(confidences, dim=1)
                entropies = softmax_entropy(final_scores)
                
                # Update caches and apply TDA
                pos_enabled = self.pos_cache_config["enabled"]
                neg_enabled = self.neg_cache_config["enabled"]
                
                # Process each box feature
                adjusted_scores = final_scores.clone()
                for i, (feat, score, pred_class, entropy) in enumerate(zip(box_features, final_scores, pred_classes, entropies)):
                    feat = feat.unsqueeze(0)  # Add batch dimension
                    norm_entropy = get_entropy(entropy, self.num_classes)
                    
                    # Update positive cache
                    if pos_enabled:
                        update_cache(self.pos_cache, pred_class.item(), [feat, entropy], self.pos_cache_config["shot_capacity"])
                    
                    # Update negative cache if entropy falls within threshold
                    if neg_enabled and (self.neg_cache_config["entropy_threshold"]["lower"] < norm_entropy < self.neg_cache_config["entropy_threshold"]["upper"]):
                        prob_map = confidences[i].unsqueeze(0)
                        update_cache(self.neg_cache, pred_class.item(), [feat, entropy, prob_map], self.neg_cache_config["shot_capacity"], True)
                
                # Compute cache adjustments
                if pos_enabled and self.pos_cache:
                    # 确保传入正确的类别数
                    actual_num_classes = adjusted_scores.size(1)  # 获取实际的类别数
                    
                    pos_logits = compute_cache_logits(
                        box_features, 
                        self.pos_cache, 
                        self.pos_cache_config["alpha"], 
                        self.pos_cache_config["beta"], 
                        actual_num_classes  # 使用实际类别数
                    )
                    
                    # 确保尺寸匹配
                if pos_logits.size(1) != adjusted_scores.size(1):
                    logger.warning(f"Size mismatch: pos_logits has {pos_logits.size(1)} classes but adjusted_scores has {adjusted_scores.size(1)} classes")
                    # 可以选择截断或者填充
                    if pos_logits.size(1) < adjusted_scores.size(1):
                        # 填充
                        padding = torch.zeros((pos_logits.size(0), adjusted_scores.size(1) - pos_logits.size(1)), 
                                            device=pos_logits.device)
                        pos_logits = torch.cat([pos_logits, padding], dim=1)
                    else:
                        # 截断
                        pos_logits = pos_logits[:, :adjusted_scores.size(1)]
                    
                        adjusted_scores += pos_logits
                
                # 对于负样本缓存的处理
                if neg_enabled and self.neg_cache:
                    # 确保使用实际的类别数
                    actual_num_classes = adjusted_scores.size(1)
                    
                    neg_logits = compute_cache_logits(
                        box_features, 
                        self.neg_cache, 
                        self.neg_cache_config["alpha"], 
                        self.neg_cache_config["beta"], 
                        actual_num_classes,  # 使用实际类别数
                        (self.neg_cache_config["mask_threshold"]["lower"], 
                        self.neg_cache_config["mask_threshold"]["upper"])
                    )
                    
                    # 确保尺寸匹配
                    if neg_logits.size(1) != adjusted_scores.size(1):
                        logger.warning(f"Size mismatch: neg_logits has {neg_logits.size(1)} classes but adjusted_scores has {adjusted_scores.size(1)} classes")
                        
                        if neg_logits.size(1) < adjusted_scores.size(1):
                            # 填充
                            padding = torch.zeros((neg_logits.size(0), adjusted_scores.size(1) - neg_logits.size(1)), 
                                                device=neg_logits.device)
                            neg_logits = torch.cat([neg_logits, padding], dim=1)
                        else:
                            # 截断
                            neg_logits = neg_logits[:, :adjusted_scores.size(1)]
                    
                    adjusted_scores -= neg_logits
                
                # Replace original scores with adjusted scores
                predictions_K[-1] = (adjusted_scores, final_deltas)
            
            # Run inference with possibly adjusted predictions
            pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                predictions_K, proposals
            )
        else:
            # Original inference path for models without refinement
            predictions = self.object_miner(box_features, proposals, context=True)
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
        
        return pred_instances, all_scores, all_boxes
    
    # Attach the TDA-enhanced method to the roi_heads
    roi_heads._forward_box = tda_forward_box.__get__(roi_heads)

if __name__ == "__main__":
    args = parse_args()
    main(args)