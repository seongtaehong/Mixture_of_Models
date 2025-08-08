import argparse
import json
import os
import glob
from pathlib import Path

models={
"1": "meta-llama__Llama-3.2-1B-Instruct",
"2": "meta-llama__Llama-3.2-3B-Instruct",
"3": "meta-llama__Llama-3.1-8B-Instruct",
"4": "mistralai__Ministral-8B-Instruct-2410",
"5": "mistralai__Mistral-7B-Instruct-v0.3",
"6": "google__gemma-2-2b-it",
"7": "google__gemma-2-9b-it",
"8": "Qwen__Qwen2.5-7B-Instruct",
"9": "google__gemma-3-1b-it",
"10": "google__gemma-3-4b-it",
"11": "deepseek-ai__deepseek-math-7b-instruct",
"12": "Qwen__Qwen2.5-Math-7B-Instruct",
"13": "nvidia__OpenMath2-Llama3.1-8B",
"14": "MathLLMs__MathCoder-CL-7B",
"15": "BioMistral__BioMistral-7B",
"16": "aaditya__Llama3-OpenBioLLM-8B",
"17": "Henrychur__MMed-Llama-3-8B",
"18": "AdaptLLM__medicine-chat",
"19": "google__medgemma-4b-it",
"20": "AdaptLLM__law-chat",
"21": "Equall__Saul-7B-Instruct-v1",
"22": "instruction-pretrain__finance-Llama3-8B",
"23": "AdaptLLM__finance-chat",
"24": "Qwen__Qwen2.5-3B-Instruct",
"25": "microsoft__Phi-3-mini-4k-instruct",
"26": "Qwen__Qwen3-1.7B",
"27": "Qwen__Qwen3-4B"
}

# 지원하는 데이터셋 목록
SUPPORTED_DATASETS = {
    "mathqa": "samples_mathqa_*.jsonl", 
    "medmcqa": "samples_medmcqa_*.jsonl",
    "csqa": "samples_commonsense_qa_*.jsonl",
    "pubmedqa": "samples_pubmedqa_*.jsonl",
    "arc": "samples_arc_*.jsonl",
    "hellaswag": "samples_hellaswag_*.jsonl",
    "logiqa": "samples_logiqa_*.jsonl",
    "mmlu": "samples_mmlu_*.jsonl"
}

# MMLU 카테고리별 서브태스크 정의
MMLU_CATEGORIES = {
    "stem": [  # 과학, 기술, 공학, 수학
        "abstract_algebra",  # 추상대수학
        "anatomy",  # 해부학
        "astronomy",  # 천문학
        "college_biology",  # 대학 생물학
        "college_chemistry",  # 대학 화학
        "college_computer_science",  # 대학 컴퓨터 과학
        "college_mathematics",  # 대학 수학
        "college_physics",  # 대학 물리학
        "computer_security",  # 컴퓨터 보안
        "conceptual_physics",  # 개념 물리학
        "electrical_engineering",  # 전기 공학
        "elementary_mathematics",  # 초등 수학
        "high_school_biology",  # 고등학교 생물학
        "high_school_chemistry",  # 고등학교 화학
        "high_school_computer_science",  # 고등학교 컴퓨터 과학
        "high_school_mathematics",  # 고등학교 수학
        "high_school_physics",  # 고등학교 물리학
        "high_school_statistics",  # 고등학교 통계학
        "machine_learning"  # 머신러닝
    ],
    "humanities": [  # 인문학
        # "formal_logic",  # 형식 논리학
        "high_school_european_history",  # 고등학교 유럽 역사
        "high_school_us_history",  # 고등학교 미국 역사
        "high_school_world_history",  # 고등학교 세계사
        "international_law",  # 국제법
        "jurisprudence",  # 법학
        "logical_fallacies",  # 논리적 오류
        "moral_disputes",  # 도덕적 논쟁
        "moral_scenarios",  # 도덕적 시나리오
        "philosophy",  # 철학
        "prehistory",  # 선사 시대
        "professional_law",  # 전문 법률
        "world_religions"  # 세계 종교
    ],
    "social_sciences": [  # 사회과학
        "econometrics",  # 계량경제학
        "high_school_geography",  # 고등학교 지리학
        "high_school_government_and_politics",  # 고등학교 정부 및 정치
        "high_school_psychology",  # 고등학교 심리학
        "human_sexuality",  # 인간 성
        "professional_psychology",  # 전문 심리학
        "public_relations",  # 홍보
        "security_studies",  # 보안 연구
        "sociology",  # 사회학
        "us_foreign_policy",  # 미국 외교 정책
        "high_school_macroeconomics",  # 고등학교 거시경제학
        "high_school_microeconomics"  # 고등학교 미시경제학
    ],
    "other": [  # 기타
        "business_ethics",  # 비즈니스 윤리
        "clinical_knowledge",  # 임상 지식
        "college_medicine",  # 대학 의학
        "global_facts",  # 국제 상식
        "human_aging",  # 인간 노화
        "management",  # 경영
        "marketing",  # 마케팅
        "medical_genetics",  # 의학 유전학
        "miscellaneous",  # 기타
        "nutrition",  # 영양학
        "professional_accounting",  # 전문 회계
        "professional_medicine",  # 전문 의학
        "virology"  # 바이러스학
    ]
}

# 기본 데이터셋 (mmlu 제외)
# DEFAULT_DATASETS = ["mathqa", "medmcqa", "csqa", "pubmedqa", "arc", "hellaswag", "logiqa"]
# DEFAULT_DATASETS = ["mathqa", "medmcqa", "pubmedqa", "arc", "logiqa"]
DEFAULT_DATASETS = ["mathqa", "pubmedqa","logiqa"]

def parse_args():
    parser = argparse.ArgumentParser(description='레이블링 데이터 생성 (전체 데이터 사용)')
    parser.add_argument('--model_ids', nargs='+', type=str, required=True, 
                        help='레이블링에 사용할 모델 번호 (예: 1 2 3)')
    parser.add_argument('--datasets', nargs='+', type=str, 
                        choices=list(SUPPORTED_DATASETS.keys()),
                        default=DEFAULT_DATASETS,
                        help=f'사용할 데이터셋 (기본값: {", ".join(DEFAULT_DATASETS)}). mmlu 포함하려면 명시적으로 지정')
    parser.add_argument('--include_mmlu', action='store_true',
                        help='mmlu 데이터셋을 기본 데이터셋에 추가')
    parser.add_argument('--mmlu_categories', nargs='+', type=str,
                        choices=['stem', 'humanities', 'social_sciences', 'other'],
                        default=['stem'],
                        help='포함할 MMLU 카테고리 (기본값: stem). 예: --mmlu_categories stem humanities')
    parser.add_argument('--base_path', type=str, default='/mnt/raid6/hst/kt/paper/eval/confidence/output',
                        help='모델 결과 JSONL 파일들이 위치한 base 경로')
    parser.add_argument('--output_dir', type=str, default='./labeled_data',
                        help='레이블링 결과 저장 경로')
    return parser.parse_args()

def should_exclude_mmlu_subtask(file_path, included_categories):
    """선택된 카테고리에 포함되지 않은 MMLU 서브태스크인지 확인합니다."""
    filename = os.path.basename(file_path)
    # samples_mmlu_formal_logic_2025-05-22T19-03-58.053555.jsonl -> formal_logic
    if filename.startswith("samples_mmlu_"):
        parts = filename.split("_")
        if len(parts) >= 3:
            subtask = "_".join(parts[2:-1])  # 타임스탬프 부분 제거
            if subtask.endswith(".jsonl"):
                subtask = subtask[:-6]
            
            # 어떤 카테고리에 속하는지 확인
            for category, subtasks in MMLU_CATEGORIES.items():
                if subtask in subtasks:
                    # 포함된 카테고리에 있는지 확인
                    return category not in included_categories
            
            # 정의되지 않은 서브태스크는 제외
            return True
    return False

def load_and_process_jsonl_files(model_dirs, base_path, datasets, mmlu_categories=None):
    """JSONL 파일을 로드하고 처리합니다."""
    all_model_data = {}
    question_answers = {}
    total_items_count = 0
    
    if mmlu_categories is None:
        mmlu_categories = ['stem']
    
    for dataset in datasets:
        print(f"\n=== {dataset} 데이터셋 처리 중 ===")
        if dataset == "mmlu":
            print(f"포함된 MMLU 카테고리: {', '.join(mmlu_categories)}")
        
        # 데이터셋별 파일 패턴 가져오기
        file_pattern = SUPPORTED_DATASETS[dataset]
        
        for model_id, model_dir in model_dirs.items():
            model_path = os.path.join(base_path, dataset, model_dir)
            if not os.path.exists(model_path):
                print(f"경로를 찾을 수 없음: {model_path}")
                continue
            
            print(f"모델 {model_id}({model_dir}) 처리 중...")
            # 데이터셋별 JSONL 파일 찾기
            jsonl_files = glob.glob(os.path.join(model_path, file_pattern))
            
            # MMLU 데이터셋의 경우 선택된 카테고리만 포함
            if dataset == "mmlu":
                original_count = len(jsonl_files)
                jsonl_files = [f for f in jsonl_files if not should_exclude_mmlu_subtask(f, mmlu_categories)]
                excluded_count = original_count - len(jsonl_files)
                print(f"- MMLU 서브태스크 필터링: {len(jsonl_files)}개 포함, {excluded_count}개 제외")
            else:
                print(f"- {len(jsonl_files)}개의 JSONL 파일 발견")
            
            model_data = []
            model_items_count = 0
            
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            doc_id = data.get("doc_id", "")
                            doc = data.get("doc", {})
                            
                            # question 추출 방식 개선
                            question = ""
                            if "arguments" in data and "gen_args_0" in data["arguments"] and "arg_0" in data["arguments"]["gen_args_0"]:
                                question = data["arguments"]["gen_args_0"]["arg_0"]
                            elif "question" in doc:
                                question = doc["question"]
                            elif "input" in doc:
                                question = doc["input"]
                            elif "ctx" in doc:  # hellaswag의 경우
                                question = doc["ctx"]
                            
                            if not question:
                                continue
                            
                            answer_idx = doc.get("answer", -1)
                            accuracy = data.get("acc", 0.0)
                            
                            # 응답에서 확신도 추출 (likelihood) - 개선된 방법
                            filtered_resps = data.get("filtered_resps", [])
                            likelihood = 0.0  # 기본값을 0.0으로 설정
                            
                            # 여러 방법으로 likelihood 추출 시도
                            if filtered_resps:
                                if isinstance(filtered_resps, list) and len(filtered_resps) > 0:
                                    # 첫 번째 요소가 숫자인 경우
                                    if isinstance(filtered_resps[0], (int, float)):
                                        likelihood = float(filtered_resps[0])
                                    # 첫 번째 요소가 리스트인 경우
                                    elif isinstance(filtered_resps[0], list) and len(filtered_resps[0]) > 0:
                                        likelihood = float(filtered_resps[0][0])
                                    # 첫 번째 요소가 딕셔너리인 경우
                                    elif isinstance(filtered_resps[0], dict):
                                        if 'logprob' in filtered_resps[0]:
                                            likelihood = float(filtered_resps[0]['logprob'])
                                        elif 'score' in filtered_resps[0]:
                                            likelihood = float(filtered_resps[0]['score'])
                            
                            # 대안으로 다른 필드에서 확신도 추출
                            if likelihood == 0.0:  # 여전히 기본값이면
                                if 'logprobs' in data:
                                    logprobs = data['logprobs']
                                    if isinstance(logprobs, list) and len(logprobs) > 0:
                                        likelihood = float(logprobs[0])
                                elif 'acc_norm' in data:
                                    likelihood = float(data.get('acc_norm', 0.0))
                                else:
                                    # 마지막 대안: 정확도를 확신도로 사용
                                    likelihood = accuracy
                            
                            # 데이터셋 정보를 포함한 고유 키 생성
                            question_key = f"{dataset}:{question}"
                            
                            # 데이터 저장
                            model_data.append({
                                "doc_id": doc_id,
                                "question": question,
                                "answer": answer_idx,
                                "accuracy": accuracy,
                                "likelihood": likelihood,
                                "dataset": dataset
                            })
                            
                            # question_answers 딕셔너리에 question 키가 없으면 초기화
                            if question_key not in question_answers:
                                question_answers[question_key] = {}
                            
                            # 해당 질문에 모델 ID와 모델명, 정확도, 확신도 정보 저장
                            question_answers[question_key][model_id] = {
                                "model_name": model_dir,
                                "accuracy": accuracy,
                                "likelihood": likelihood,
                                "answer": answer_idx,
                                "dataset": dataset
                            }
                            
                            model_items_count += 1
                            
                except json.JSONDecodeError:
                    print(f"JSON 파싱 오류: {line}")
                except Exception as e:
                    print(f"처리 오류: {e}")
            
            if model_id not in all_model_data:
                all_model_data[model_id] = []
            all_model_data[model_id].extend(model_data)
            print(f"- 모델 {model_id}에서 처리된 항목 수: {model_items_count}")
            total_items_count += model_items_count
    
    print(f"\n총 처리된 데이터셋 항목 수: {total_items_count}")
    return all_model_data, question_answers

def create_labeling_data(question_answers):
    """레이블링 데이터 생성"""
    labeling_data = []
    
    print(f"총 {len(question_answers)}개의 질문 처리 중...")
    
    # 디버깅을 위한 통계
    accuracy_stats = {"1.0": 0, "0.0": 0, "other": 0}
    models_with_correct_answers = set()
    
    for question_key, models_info in question_answers.items():
        # 각 모델의 정확도 통계 수집
        for model_id, info in models_info.items():
            acc = info['accuracy']
            if acc == 1.0:
                accuracy_stats["1.0"] += 1
                models_with_correct_answers.add(model_id)
            elif acc == 0.0:
                accuracy_stats["0.0"] += 1
            else:
                accuracy_stats["other"] += 1
        
        # acc가 1.0인 모델들 찾기
        correct_models = {model_id: info for model_id, info in models_info.items() 
                          if info['accuracy'] == 1.0}
        
        if correct_models:
            # acc가 1.0인 모델이 여러 개 있는 경우, 확신도(likelihood)가 가장 높은 모델 선택
            max_likelihood = float("-inf")
            selected_model_id = None
            
            for model_id, info in correct_models.items():
                if info['likelihood'] > max_likelihood:
                    max_likelihood = info['likelihood']
                    selected_model_id = model_id
            
            if selected_model_id:
                # 모델 ID 대신 모델명을 레이블로 사용
                selected_model_name = correct_models[selected_model_id]["model_name"]
                dataset = correct_models[selected_model_id]["dataset"]
                
                # 데이터셋 접두사 제거하여 원래 질문만 추출
                question = question_key.split(":", 1)[1]
                
                # 레이블링 데이터에 추가
                labeling_data.append({
                    "question": question, 
                    "label": selected_model_name,
                    "dataset": dataset
                })
    
    # 디버깅 정보 출력
    print(f"\n=== 디버깅 정보 ===")
    print(f"정확도 1.0인 답변: {accuracy_stats['1.0']}개")
    print(f"정확도 0.0인 답변: {accuracy_stats['0.0']}개")
    print(f"기타 정확도 답변: {accuracy_stats['other']}개")
    print(f"정답을 맞춘 모델들: {sorted(models_with_correct_answers)}")
    
    # 샘플 데이터 출력 (처음 3개)
    sample_count = 0
    for question_key, models_info in question_answers.items():
        if sample_count >= 3:
            break
        print(f"\n샘플 질문 {sample_count + 1}:")
        print(f"질문: {question_key[:100]}...")
        for model_id, info in models_info.items():
            print(f"  모델 {model_id}: 정확도={info['accuracy']}, 확신도={info['likelihood']:.4f}")
        sample_count += 1
    
    print(f"\n레이블링 데이터 생성 완료: {len(labeling_data)}개 항목")
    return labeling_data

def save_labeling_data(labeling_data, output_dir="./", model_ids=None, datasets=None):
    """레이블링 데이터를 하나의 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터셋명과 모델 ID를 포함한 파일명 생성
    if model_ids and datasets:
        dataset_str = "_".join(sorted(datasets))
        model_filename = f'{dataset_str}_{"_".join(model_ids)}_full.jsonl'
    elif datasets:
        dataset_str = "_".join(sorted(datasets))
        model_filename = f'{dataset_str}_all_models_full.jsonl'
    else:
        model_filename = 'all_models_full.jsonl'
    
    model_filename = f'{"_".join(model_ids)}.jsonl'
    output_path = os.path.join(output_dir, model_filename)
    
    # 하나의 파일에 모든 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in labeling_data:
            question = item["question"]
            dataset = item["dataset"]
            
            # mmlu 데이터셋의 경우 기존 처리 방식 유지
            if dataset == "mmlu" and '\n\n' in question:
                question_parts = question.split('\n\n')
                if len(question_parts) > 1:
                    question = question_parts[1]
            
            json_line = json.dumps({
                "question": question, 
                "label": item["label"],
                "dataset": dataset
            }, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"파일 저장 완료: {output_path} (항목 수: {len(labeling_data)})")
    
    # 데이터셋별 통계 출력
    dataset_counts = {}
    for item in labeling_data:
        dataset = item["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    print("\n데이터셋별 항목 수:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"- {dataset}: {count}개")
    
    # 레이블별 통계 출력
    label_counts = {}
    for item in labeling_data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n레이블별 항목 수:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {count}개")
    
    # 데이터셋별 레이블 분포 출력
    dataset_label_counts = {}
    for item in labeling_data:
        dataset = item["dataset"]
        label = item["label"]
        if dataset not in dataset_label_counts:
            dataset_label_counts[dataset] = {}
        dataset_label_counts[dataset][label] = dataset_label_counts[dataset].get(label, 0) + 1
    
    print("\n데이터셋별 레이블 분포:")
    for dataset, label_dist in sorted(dataset_label_counts.items()):
        print(f"\n{dataset}:")
        for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {label}: {count}개")

def main():
    args = parse_args()
    model_ids = args.model_ids
    datasets = args.datasets
    
    # --include_mmlu 플래그가 설정된 경우 mmlu를 데이터셋에 추가
    if args.include_mmlu and "mmlu" not in datasets:
        datasets.append("mmlu")
    
    base_path = args.base_path
    
    # 선택된 모델 ID와 이름 매핑 가져오기
    model_dirs = {model_id: models[model_id] for model_id in model_ids if model_id in models}
    
    if not model_dirs:
        print("선택한 모델 ID가 올바르지 않습니다.")
        return
    
    print(f"선택된 데이터셋: {datasets}")
    print(f"선택된 모델: {model_dirs}")
    print("*** 균형 조정 없이 모든 데이터를 사용합니다 ***")
    
    # JSONL 파일 로드 및 처리
    all_model_data, question_answers = load_and_process_jsonl_files(model_dirs, base_path, datasets, args.mmlu_categories)
    
    if not question_answers:
        print("처리할 데이터가 없습니다.")
        return
    
    # 레이블링 데이터 생성 (균형 조정 없음)
    labeling_data = create_labeling_data(question_answers)
    
    # 데이터셋별 레이블 분포 출력
    print("\n=== 최종 레이블 분포 (균형 조정 없음) ===")
    dataset_groups = {}
    for item in labeling_data:
        dataset = item["dataset"]
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(item)
    
    for dataset, items in dataset_groups.items():
        # 현재 데이터셋의 레이블별 개수 계산
        label_counts = {}
        for item in items:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n{dataset} 데이터셋 레이블 분포:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {label}: {count}개")
    
    print(f"\n전체 데이터 항목 수: {len(labeling_data)}개")
    
    # 레이블링 데이터 저장
    output_dir = args.output_dir
    save_labeling_data(labeling_data, output_dir, model_ids, datasets)

if __name__ == "__main__":
    main()
    
    

# python make_label_full.py --model_ids 12 16 19 --include_mmlu --mmlu_categories social_sciences other
# python3 make_label_full.py --model_ids 24 25 --datasets medmcqa arc csqa mathqa logiqa  --include_mmlu --mmlu_categories social_sciences humanities stem other

