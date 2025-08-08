import argparse
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example usage:
# python inference_pipeline.py --orchestrator_path "./ckpt/modern" --query "What is machine learning?"

class MoMInferencePipeline:
    """
    Main inference pipeline for the MoM framework.
    
    This class handles loading the orchestrator model and routing queries
    to the appropriate specialized models based on orchestrator predictions.
    """
    
    def __init__(self, orchestrator_path: str, available_models: Dict[str, str]):
        """
        Initialize the MoM inference pipeline.
        
        Args:
            orchestrator_path (str): Path to the trained orchestrator model
            available_models (Dict[str, str]): Mapping of model names to their HuggingFace model IDs
        """
        self.orchestrator_path = orchestrator_path
        self.available_models = available_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load orchestrator
        self.orchestrator_tokenizer = None
        self.orchestrator_model = None
        self.id2label = None
        self._load_orchestrator()
        
        # Cache for loaded models to avoid reloading
        self.loaded_models = {}
        
        logger.info(f"MoM Pipeline initialized with {len(available_models)} available models")
    
    def _load_orchestrator(self):
        """
        Load the trained orchestrator model and tokenizer.
        """
        logger.info(f"Loading orchestrator from: {self.orchestrator_path}")
        
        try:
            self.orchestrator_tokenizer = AutoTokenizer.from_pretrained(self.orchestrator_path)
            self.orchestrator_model = AutoModelForSequenceClassification.from_pretrained(
                self.orchestrator_path
            )
            self.orchestrator_model.to(self.device)
            self.orchestrator_model.eval()
            
            # Get label mappings
            self.id2label = self.orchestrator_model.config.id2label
            
            logger.info("Orchestrator loaded successfully")
            logger.info(f"Available model classes: {list(self.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load orchestrator: {e}")
            raise
    
    def predict_best_model(self, query: str) -> str:
        """
        Use the orchestrator to predict which model should handle the query.
        
        Args:
            query (str): User input query
            
        Returns:
            str: Name of the predicted best model
        """
        # Tokenize input
        inputs = self.orchestrator_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.orchestrator_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
        
        # Convert to model name
        predicted_model = self.id2label[predicted_class_id]
        confidence = predictions[0][predicted_class_id].item()
        
        # Debug: Print all possible predictions with confidence scores
        logger.info("All model predictions:")
        for i, (label_id, label_name) in enumerate(self.id2label.items()):
            conf = predictions[0][i].item()
            logger.info(f"  {label_name}: {conf:.4f}")
        
        logger.info(f"Orchestrator prediction: '{predicted_model}' (confidence: {confidence:.4f})")
        
        return predicted_model
    
    def _load_selected_model(self, model_name: str):
        """
        Load the selected model for text generation.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            pipeline: HuggingFace text generation pipeline
        """
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not found in available models")
            return None
        
        model_id = self.available_models[model_name]
        logger.info(f"Loading model: {model_name} ({model_id})")
        
        try:
            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            # Cache the loaded model
            self.loaded_models[model_name] = text_generator
            logger.info(f"Successfully loaded model: {model_name}")
            
            return text_generator
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def generate_response(self, query: str, selected_model: str) -> str:
        """
        Generate response using the selected model.
        
        Args:
            query (str): User input query
            selected_model (str): Name of the model to use
            
        Returns:
            str: Generated response
        """
        # Load the selected model
        text_generator = self._load_selected_model(selected_model)
        
        if text_generator is None:
            logger.warning(f"Model {selected_model} not available, using default response")
            return f"I apologize, but the selected model '{selected_model}' is not currently available."
        
        try:
            # Generate response
            logger.info(f"Generating response with {selected_model}")
            
            # Format the prompt (you can customize this based on your needs)
            prompt = f"Question: {query}\nAnswer:"
            
            # Generate text
            outputs = text_generator(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=text_generator.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the original prompt from the response
            response = generated_text[len(prompt):].strip()
            
            logger.info(f"Response generated successfully with {selected_model}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {selected_model}: {e}")
            return f"Sorry, I encountered an error while generating a response with {selected_model}."
    
    def process_query(self, query: str) -> Dict[str, str]:
        """
        Complete pipeline: predict best model and generate response.
        
        Args:
            query (str): User input query
            
        Returns:
            Dict[str, str]: Dictionary containing selected model and response
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Predict best model
        selected_model = self.predict_best_model(query)
        
        # Step 2: Generate response with selected model
        response = self.generate_response(query, selected_model)
        
        return {
            "query": query,
            "selected_model": selected_model,
            "response": response
        }


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="MoM Framework Inference Pipeline")
    parser.add_argument(
        "--orchestrator_path",
        type=str,
        required=True,
        help="Path to the trained orchestrator model"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query to process"
    )
    
    args = parser.parse_args()
    
    # Define available models (model_name -> HuggingFace model ID)
    available_models = {
        "meta-llama__Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama__Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama__Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai__Ministral-8B-Instruct-2410": "mistralai/Ministral-8B-Instruct-2410",
        "mistralai__Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "google__gemma-2-2b-it": "google/gemma-2-2b-it",
        "google__gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen__Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "google__gemma-3-1b-it": "google/gemma-3-1b-it",
        "google__gemma-3-4b-it": "google/gemma-3-4b-it",
        "deepseek-ai__deepseek-math-7b-instruct": "deepseek-ai/deepseek-math-7b-instruct",
        "Qwen__Qwen2.5-Math-7B-Instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
        "nvidia__OpenMath2-Llama3.1-8B": "nvidia/OpenMath2-Llama3.1-8B",
        "MathLLMs__MathCoder-CL-7B": "MathLLMs/MathCoder-CL-7B",
        "BioMistral__BioMistral-7B": "BioMistral/BioMistral-7B",
        "aaditya__Llama3-OpenBioLLM-8B": "aaditya/Llama3-OpenBioLLM-8B",
        "Henrychur__MMed-Llama-3-8B": "Henrychur/MMed-Llama-3-8B",
        "AdaptLLM__medicine-chat": "AdaptLLM/medicine-chat",
        "google__medgemma-4b-it": "google/medgemma-4b-it",
        "AdaptLLM__law-chat": "AdaptLLM/law-chat",
        "Equall__Saul-7B-Instruct-v1": "Equall/Saul-7B-Instruct-v1",
        "instruction-pretrain__finance-Llama3-8B": "instruction-pretrain/finance-Llama3-8B",
        "AdaptLLM__finance-chat": "AdaptLLM/finance-chat",
        "Qwen__Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        "microsoft__Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
        "Qwen__Qwen3-1.7B": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen__Qwen3-4B": "Qwen/Qwen2.5-3B-Instruct"
    }
    
    # Initialize pipeline
    pipeline = MoMInferencePipeline(
        orchestrator_path=args.orchestrator_path,
        available_models=available_models
    )
    
    # Process query
    result = pipeline.process_query(args.query)
    
    # Display results
    print("\n" + "="*50)
    print("MoM Framework Inference Results")
    print("="*50)
    print(f"Query: {result['query']}")
    print(f"Selected Model: {result['selected_model']}")
    print(f"Response: {result['response']}")
    print("="*50)


if __name__ == "__main__":
    main()



