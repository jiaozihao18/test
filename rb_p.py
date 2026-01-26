"""
序列推荐模型预测脚本
用于加载训练好的 RecBole 模型并预测下一个 item
"""

import torch
import csv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model


class ModelPredictor:
    """模型预测器类，封装模型加载和预测逻辑"""
    
    def __init__(self, model_file: str):
        """初始化预测器，加载模型和数据"""
        self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = \
            load_data_and_model(model_file=model_file)
        self.model.eval()
        self.max_seq_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.device = self.config["device"]
        
        # 从模型中获取字段名（更可靠）
        self.item_seq_field = self.model.ITEM_SEQ  # 例如: "item_id_list"
        self.item_seq_len_field = self.model.ITEM_SEQ_LEN  # 例如: "item_length"
        self.iid_field = self.dataset.iid_field  # item_id字段名
    
    def _convert_to_internal_ids(self, item_id_list: List[Any]) -> List[int]:
        """将外部item_id转换为内部token ID（使用RecBole的批量转换）"""
        try:
            # token2id支持列表输入，会自动批量转换
            internal_ids = self.dataset.token2id(self.iid_field, item_id_list)
            # 过滤掉无效的ID（如果token不在数据集中会抛出异常）
            if isinstance(internal_ids, np.ndarray):
                return internal_ids.tolist()
            return internal_ids if isinstance(internal_ids, list) else [internal_ids]
        except (ValueError, TypeError):
            # 如果批量转换失败，逐个转换并过滤无效项
            internal_ids = []
            for item_id in item_id_list:
                try:
                    internal_id = self.dataset.token2id(self.iid_field, item_id)
                    internal_ids.append(internal_id)
                except ValueError:
                    continue
            return internal_ids
    
    def _convert_to_external_ids(self, internal_ids: List[int]) -> List[Any]:
        """将内部token ID转换为外部item_id（使用RecBole的批量转换）"""
        try:
            # id2token支持numpy数组和列表，可以直接批量转换
            ids_array = np.array(internal_ids)
            external_ids = self.dataset.id2token(self.iid_field, ids_array)
            if isinstance(external_ids, np.ndarray):
                return external_ids.tolist()
            return external_ids if isinstance(external_ids, list) else [external_ids]
        except (KeyError, IndexError, ValueError):
            # 如果批量转换失败，逐个转换
            external_ids = []
            for idx in internal_ids:
                try:
                    external_id = self.dataset.id2token(self.iid_field, idx)
                    external_ids.append(external_id)
                except (KeyError, IndexError):
                    continue
            return external_ids
    
    def _prepare_sequence(self, internal_ids: List[int]) -> Tuple[List[int], int]:
        """准备序列：处理长度和padding"""
        if len(internal_ids) == 0:
            return [0] * self.max_seq_len, 1
        
        if len(internal_ids) > self.max_seq_len:
            internal_ids = internal_ids[-self.max_seq_len:]
            seq_len = self.max_seq_len
        else:
            seq_len = len(internal_ids)
        
        padded_seq = [0] * self.max_seq_len
        padded_seq[:seq_len] = internal_ids
        return padded_seq, seq_len
    
    def _create_interaction(self, sequences: List[List[int]], lengths: List[int]) -> Interaction:
        """创建Interaction对象（使用模型中的字段名）"""
        interaction_dict = {
            self.item_seq_field: torch.tensor(sequences, dtype=torch.long),
            self.item_seq_len_field: torch.tensor(lengths, dtype=torch.long)
        }
        return Interaction(interaction_dict).to(self.device)
    
    def predict_single(self, item_id_list: List[Any], topk: int = 10) -> Tuple[List[Any], List[float]]:
        """预测单个序列"""
        internal_ids = self._convert_to_internal_ids(item_id_list)
        if len(internal_ids) == 0:
            raise ValueError("No valid item_id found in the input sequence")
        
        padded_seq, seq_len = self._prepare_sequence(internal_ids)
        interaction = self._create_interaction([padded_seq], [seq_len])
        
        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction).squeeze(0)
            scores[0] = -float('inf')  # mask padding
            topk_scores, topk_indices = torch.topk(scores, min(topk, len(scores)))
        
        external_indices = topk_indices.cpu().numpy().tolist()
        external_items = self._convert_to_external_ids(external_indices)
        external_scores = topk_scores.cpu().numpy().tolist()
        
        return external_items, external_scores
    
    def predict_batch(self, item_id_lists: List[List[Any]], topk: int = 10, 
                      batch_size: int = 64, show_progress: bool = False) -> List[List[Tuple[Any, float]]]:
        """批量预测多个序列"""
        results = []
        total = len(item_id_lists)
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_seqs = item_id_lists[batch_start:batch_end]
            
            # 准备批次数据
            batch_sequences = []
            batch_lengths = []
            
            for item_id_list in batch_seqs:
                internal_ids = self._convert_to_internal_ids(item_id_list)
                padded_seq, seq_len = self._prepare_sequence(internal_ids)
                batch_sequences.append(padded_seq)
                batch_lengths.append(seq_len)
            
            # 创建interaction并预测
            interaction = self._create_interaction(batch_sequences, batch_lengths)
            
            with torch.no_grad():
                scores = self.model.full_sort_predict(interaction)
                scores[:, 0] = -float('inf')  # mask padding
                topk_scores, topk_indices = torch.topk(scores, topk, dim=1)
            
            # 批量转换结果（更高效）
            batch_indices = topk_indices.cpu().numpy()  # [batch_size, topk]
            batch_scores = topk_scores.cpu().numpy()
            
            # 转换回外部 ID
            for i in range(len(batch_seqs)):
                external_indices = batch_indices[i].tolist()
                external_items = self._convert_to_external_ids(external_indices)
                external_scores = batch_scores[i].tolist()
                results.append(list(zip(external_items, external_scores)))
            
            if show_progress and (batch_end % (batch_size * 10) == 0 or batch_end == total):
                print(f"已处理 {batch_end}/{total} 个样本...")
        
        return results


def write_results_to_csv(results: List[Dict[str, Any]], output_file: str, 
                         columns: Optional[List[str]] = None) -> None:
    """将预测结果写入CSV文件"""
    if len(results) == 0:
        print("警告: 结果列表为空，不写入文件")
        return
    
    if columns is None:
        columns = list(results[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for result in results:
            writer.writerow([result.get(col, '') for col in columns])


def example1_single_sequence(input_sequence: List[Any], model_file: str, 
                             topk: int = 10, output_file: Optional[str] = None) -> Optional[Tuple[List[Any], List[float]]]:
    """示例1: 单个序列预测并可选写入CSV"""
    print("=" * 60)
    print("示例 1: 单个序列预测")
    print("=" * 60)
    
    try:
        predictor = ModelPredictor(model_file)
        top_items, scores = predictor.predict_single(input_sequence, topk)
        
        print(f"\n输入序列: {input_sequence}")
        print(f"\nTop-{len(top_items)} 推荐结果:")
        print("-" * 60)
        for i, (item, score) in enumerate(zip(top_items, scores), 1):
            print(f"{i:2d}. Item ID: {item:8s}, Score: {score:8.4f}")
        
        if output_file:
            results = [{
                '历史序列': ','.join(map(str, input_sequence)),
                'Top预测': ','.join(map(str, top_items)),
                '分数': ','.join(f"{s:.4f}" for s in scores)
            }]
            write_results_to_csv(results, output_file)
            print(f"\n结果已保存到: {output_file}")
        
        return top_items, scores
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def example2_batch_sequences(sequences: List[List[Any]], model_file: str, 
                             topk: int = 5, output_file: Optional[str] = None, 
                             batch_size: int = 64) -> Optional[List[List[Tuple[Any, float]]]]:
    """示例2: 批量预测多个序列并可选写入CSV"""
    print("=" * 60)
    print("示例 2: 批量预测多个序列")
    print("=" * 60)
    
    try:
        predictor = ModelPredictor(model_file)
        batch_results = predictor.predict_batch(sequences, topk, batch_size, show_progress=True)
        
        # 打印结果
        for idx, (seq, results) in enumerate(zip(sequences, batch_results), 1):
            print(f"\n序列 {idx}: {seq}")
            print(f"Top-{len(results)} 推荐:")
            for i, (item, score) in enumerate(results, 1):
                print(f"  {i}. Item ID: {item:8s}, Score: {score:8.4f}")
        
        if output_file:
            csv_results = [
                {
                    '历史序列': ','.join(map(str, seq)),
                    'Top预测': ','.join(str(item) for item, _ in results)
                }
                for seq, results in zip(sequences, batch_results)
            ]
            write_results_to_csv(csv_results, output_file)
            print(f"\n结果已保存到: {output_file}")
        
        return batch_results
    except Exception as e:
        print(f"批量预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def example3_dataset_predict(model_file: str, num_samples: Optional[int] = 100, 
                             topk: int = 5, batch_size: int = 64, 
                             output_file: str = "prediction_results.csv") -> List[Dict[str, Any]]:
    """示例3: 从数据集获取真实序列进行预测并写入CSV（批量处理加速）"""
    print("=" * 60)
    print("示例 3: 从测试集获取序列进行预测并写入文件（批量处理）")
    print("=" * 60)
    
    try:
        predictor = ModelPredictor(model_file)
        
        if len(predictor.test_data.dataset) == 0:
            print("测试集为空，无法获取示例序列")
            return []
        
        # 确定要处理的样本数量
        total_samples = len(predictor.test_data.dataset)
        num_samples = total_samples if num_samples is None else min(num_samples, total_samples)
        
        print(f"\n处理 {num_samples} 个样本（批大小: {batch_size}）...")
        
        # 收集所有样本的外部ID序列（使用模型中的字段名）
        external_sequences = []
        for idx in range(num_samples):
            sample = predictor.test_data.dataset[idx]
            if predictor.item_seq_field in sample:
                item_seq = sample[predictor.item_seq_field]
                item_length = sample[predictor.item_seq_len_field].item()
                
                # 批量转换为外部ID（更高效）
                internal_ids = item_seq[:item_length].cpu().numpy().tolist()
                try:
                    external_seq = predictor._convert_to_external_ids(internal_ids)
                    if len(external_seq) > 0:
                        external_sequences.append(external_seq)
                except:
                    continue
        
        print(f"收集到 {len(external_sequences)} 个有效序列，开始批量预测...")
        
        # 批量预测
        batch_results = predictor.predict_batch(
            external_sequences, topk, batch_size, show_progress=True
        )
        
        # 格式化结果
        results = [
            {
                '历史序列': ','.join(map(str, external_seq)),
                f'Top{topk}预测': ','.join(str(item) for item, _ in predictions[:topk])
            }
            for external_seq, predictions in zip(external_sequences, batch_results)
        ]
        
        # 写入CSV文件
        write_results_to_csv(results, output_file)
        
        print(f"\n预测完成！结果已保存到: {output_file}")
        print(f"共处理 {len(results)} 个样本")
        print(f"\n前5个结果预览:")
        print("-" * 60)
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. 历史序列: {result['历史序列']}")
            print(f"   Top{topk}预测: {result[f'Top{topk}预测']}")
            print()
        
        return results
    except Exception as e:
        print(f"从测试集预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """主函数 - 使用示例"""
    model_file = "/home/j00960957/j00960957/llm4rec_add_general/recbole_eval/saved/BERT4Rec-Jan-11-2026_09-43-15.pth"
    
    """
    # 示例 1: 单个序列预测（可选写入CSV）
    example1_single_sequence(
        input_sequence=[1, 2, 3, 4, 5],
        model_file=model_file,
        topk=10,
        output_file="example1_results.csv"  # 可选，设为None则不写入文件
    )
    
    # 示例 2: 批量预测（可选写入CSV）
    example2_batch_sequences(
        sequences=[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]],
        model_file=model_file,
        topk=5,
        output_file="example2_results.csv",  # 可选，设为None则不写入文件
        batch_size=64
    )
    """
    # 示例 3: 从数据集预测并写入CSV
    example3_dataset_predict(
        model_file=model_file,
        num_samples=None,  # 处理的样本数量，None表示处理全部
        topk=5,
        batch_size=4096,
        output_file="BERT4Rec_results.csv"
    )


if __name__ == "__main__":
    main()
