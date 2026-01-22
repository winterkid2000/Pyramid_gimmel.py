class BioMistralReportGenerator:
    
    def __init__(self, model_name: str = "BioMistral/BioMistral-7B", dictionary_path: str = r'c:\Users\RaPhyA\Desktop\Nous\assets\word_dictionary.json'):
  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # Feature dictionary 로드
        self.feature_dictionary = self.load_dictionary(dictionary_path)
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto", 
            offload_folder=r"c:\Users\RaPhyA\Desktop\새 폴더 (2)",
            low_cpu_mem_usage=True
        )
    
    def load_dictionary(self, dictionary_path: str = None) -> Dict[str, str]:
        """
        word_dictionary.json 파일을 로드
        
        Args:
            dictionary_path: JSON 파일 경로 (None이면 기본 경로 사용)
        
        Returns:
            Feature 이름과 설명이 담긴 딕셔너리
        """
        if dictionary_path is None:
            # 기본 경로: 현재 스크립트와 같은 폴더
            dictionary_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                'word_dictionary.json'
            )
        
        if not os.path.exists(dictionary_path):
            print(f"⚠️ Dictionary not found: {dictionary_path}")
            return {}
        
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            print(f"✅ Loaded {len(dictionary)} feature definitions from dictionary")
            return dictionary
        except Exception as e:
            print(f"❌ Error loading dictionary: {e}")
            return {}
    
    def shap_df_to_text(self, shap_df: pd.DataFrame, top_n: int = 10, include_definitions: bool = True) -> str:
        """
        SHAP DataFrame을 텍스트로 변환하며, feature_dictionary의 설명 포함
        
        Args:
            shap_df: SHAP values를 포함한 DataFrame
            top_n: 상위 몇 개의 feature를 사용할지
            include_definitions: dictionary의 설명을 포함할지 여부
        
        Returns:
            텍스트로 변환된 SHAP 요약
        """
        # SHAP 값 기준으로 정렬 (절대값)
        if 'shap_value' in shap_df.columns:
            shap_sorted = shap_df.copy()
            shap_sorted['abs_shap'] = shap_sorted['shap_value'].abs()
            shap_sorted = shap_sorted.sort_values('abs_shap', ascending=False).head(top_n)
        else:
            shap_sorted = shap_df.head(top_n)
        
        # 텍스트 변환
        text = "SHAP Feature Importance Summary:\n\n"
        for idx, row in shap_sorted.iterrows():
            feature_name = row.get('feature', row.get('Feature', str(idx)))
            shap_val = row.get('shap_value', row.get('SHAP_value', 'N/A'))
            
            if isinstance(shap_val, (int, float)):
                direction = "positive" if shap_val > 0 else "negative"
                text += f"- {feature_name}: SHAP value = {shap_val:.4f} ({direction} contribution)\n"
                
                # Dictionary에서 설명 추가
                if include_definitions and feature_name in self.feature_dictionary:
                    definition = self.feature_dictionary[feature_name]
                    text += f"  → Definition: {definition}\n"
            else:
                text += f"- {feature_name}: {shap_val}\n"
        
        return text
    
    def generate_report(
        self, 
        shap_df: pd.DataFrame,
        feature_dictionary: Optional[Dict[str, str]] = None,
        top_n: int = 10,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        log_callback: Optional[callable] = None
    ) -> str:
        """
        SHAP DataFrame으로부터 방사선학 리포트 생성
        
        Args:
            shap_df: SHAP values를 포함한 DataFrame
            feature_dictionary: 외부에서 전달된 용어 사전 (선택, 없으면 self.feature_dictionary 사용)
            top_n: 상위 몇 개의 feature를 사용할지
            max_tokens: 생성할 최대 토큰 수
            temperature: 생성 온도 (높을수록 창의적)
        
        Returns:
            생성된 방사선학 리포트
        """
        # 외부 dictionary가 없으면 내부 dictionary 사용
        if feature_dictionary is None:
            feature_dictionary = self.feature_dictionary
        
        # SHAP 데이터를 텍스트로 변환 (정의 포함)
        shap_text = self.shap_df_to_text(shap_df, top_n, include_definitions=True)
        
        # 추가 Feature dictionary 정보 (필요시)
        dict_text = ""
        # 이미 shap_df_to_text에서 정의를 포함했으므로, 여기서는 생략하거나
        # 추가적인 일반적 설명을 넣을 수 있음
        
        # 프롬프트 구성 (더 상세하고 풍부한 리포트를 위한 프롬프트)
        prompt = f"""You are a professional professor in radiology and an expert in radiomics analysis. Based on the following SHAP analysis results with detailed feature definitions, generate a comprehensive radiological interpretation report for detecting Pancreatic Disease.

{shap_text}

Please provide a detailed report with the following sections:

0. Skip the definition of Radiomics or something that isn't unnecessary for interpretation which spents the tokens.

1. **Feature Summary**: Summarize the most important radiomics features and their SHAP values. Explain what these features represent in radiological terms using the provided definitions.

2. **Clinical Interpretation**: Provide a detailed clinical interpretation of these features. Explain:
   - What each feature indicates about the tissue characteristics
   - How these features relate to normal vs abnormal pancreatic tissue
   - The significance of positive vs negative SHAP contributions

3. **Diagnostic Implications**: Discuss:
   - What these radiomics patterns suggest about the patient's condition
   - Any specific imaging characteristics that support the diagnosis
   - Confidence level in the analysis

4. **Clinical Reasoning**: Explain in detail why the patient appears to be Normal, based on:
   - The combination of radiomics features
   - The balance of positive and negative contributions
   - Typical patterns seen in normal pancreatic tissue

5. **Final Diagnosis**: Conclude with a clear statement: 'Final Diagnosis: Normal' or 'Final Diagnosis: Abnormal'

Generate a comprehensive, medically sound report that would be useful for clinical decision-making.

Report:"""
        
        # 메시지 형식으로 변환
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 토크나이저로 입력 준비
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # 텍스트 생성
        print("Generating radiological report...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 결과 디코딩
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def batch_generate_reports(
        self,
        shap_df_list: List[pd.DataFrame],
        feature_dictionary: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[str]:
 
        reports = []
        for i, shap_df in enumerate(shap_df_list):
            print(f"\nGenerating report {i+1}/{len(shap_df_list)}...")
            report = self.generate_report(shap_df, feature_dictionary, **kwargs)
            reports.append(report)
        
        return reports
