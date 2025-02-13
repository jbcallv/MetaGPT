"""
Filename: MetaGPT/metagpt/provider/human_provider.py
Created Date: Wednesday, November 8th 2023, 11:55:46 pm
Author: garylin2099
"""
from typing import Optional

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import LLM_API_TIMEOUT, USE_CONFIG_TIMEOUT
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.llm_provider_registry import register_provider


@register_provider(LLMType.CODESTALLATION)
class CodestallationLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        """Initialize the provider with config and load model
        
        Args:
            config (LLMConfig): Configuration including model path and parameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer into memory"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self.config.model  # Use model field to store path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _generate(self, messages: list[dict], **kwargs) -> str:
        """Generate text using the loaded model
        
        Args:
            messages (list[dict]): List of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
        """
        try:
            prompt = self.messages_to_prompt(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response to match API behavior
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """Implementation of abstract method for chat completion
        
        Args:
            messages (list[dict]): List of message dictionaries
            timeout (int, optional): Timeout in seconds
            
        Returns:
            dict: Response in OpenAI-like format
        """
        response = self._generate(messages)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """Implementation of abstract method for completion
        
        Args:
            messages (list[dict]): List of message dictionaries
            timeout (int, optional): Timeout in seconds
            
        Returns:
            dict: Response in OpenAI-like format
        """
        return await self._achat_completion(messages, timeout)

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        """Implementation of abstract method for streaming chat completion
        
        Currently returns full response as streaming isn't implemented
        
        Args:
            messages (list[dict]): List of message dictionaries
            timeout (int, optional): Timeout in seconds
            
        Returns:
            str: Generated response
        """
        response = await self._achat_completion(messages, timeout)
        return self.get_choice_text(response)

    def get_choice_text(self, rsp: dict) -> str:
        """Extract text from response format
        
        Args:
            rsp (dict): Response dictionary
            
        Returns:
            str: Extracted text
        """
        return rsp["choices"][0]["message"]["content"]