import tiktoken, warnings
from transformers import AutoTokenizer, AutoConfig

class BasePromptTemplate:
    def __init__(self, config):
        self.config = config
        self.is_openai = config["framework"] == "openai"
        self.max_input_len = config['generator_max_input_len']
        if not self.is_openai:
            self.generator_path = config["generator_model_path"]
            model_config = AutoConfig.from_pretrained(self.generator_path, trust_remote_code=True)
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if "chat" in model_name or "instruct" in model_name or "deepseek" in model_name:
                self.is_chat = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
        else:
            self.is_chat = True
            self.enable_chat = True
            try:
                self.tokenizer = tiktoken.encoding_for_model(config['generator_model'])
            except Exception as e:
                print("Error: ", e)
                warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
                self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    def truncate_prompt(self, prompt):
        if self.is_openai:
            truncated_messages = []
            total_tokens = 0
            assert isinstance(prompt, list)
            for message in prompt:
                role_content = message['content']
                encoded_message = self.tokenizer.encode(role_content)

                if total_tokens + len(encoded_message) <= self.max_input_len:
                    truncated_messages.append(message)
                    total_tokens += len(encoded_message)
                else:
                    print(f"The input text length is greater than the maximum length ({total_tokens + len(encoded_message)} > {self.max_input_len}) and has been truncated!")
                    remaining_tokens = self.max_input_len - total_tokens
                    truncated_message = self.encoding.decode(encoded_message[:remaining_tokens])
                    message['content'] = truncated_message
                    truncated_messages.append(message)
                    break

            return truncated_messages

        else:
            assert isinstance(prompt, str)
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > self.max_input_len:
                print(f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!")
                half = int(self.max_input_len / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                        self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt
    
class RationalePromptTemplate(BasePromptTemplate):
    placeholders = ["query", "gold", "text"]
    
    base_system_prompt_rationale = (
        "Read the following documents relevant to the given question: {query}"
        "\n{text}"
    )
    base_user_prompt = (
        "\nPlease identify documents that are useful to answer the given question: '{query}', and explain how the contents lead to the answer.\n\n"
    )

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):
        super().__init__(config)

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt_rationale
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
    
    def format_reference(self, passages):
        format_reference = ""
        if passages is None:
            return format_reference
            
        for idx, content in enumerate(passages):
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def get_string(self, query=None, text=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    return self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        input_params = {"query": query, "text": text}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        self.is_chat = True
        if self.is_chat and self.enable_chat:
            # input = []
            # if system_prompt != "":
            #     input.append({"role": "system", "content": system_prompt})
            # if user_prompt != "":
            #     input.append({"role": "user", "content": user_prompt})
            # if not self.is_openai:
            #     input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            # return self.truncate_prompt(input)
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])
            return self.truncate_prompt(input)
        

class CritiqueRationalePromptTemplate(BasePromptTemplate):
    placeholders = ["query", "gold", "text", "rationale"]

    base_system_prompt_critique = (
        "Read the following documents relevant to the given question: {query}"
        "{text}"
    )
    base_user_prompt = (
        "\nHere is the given weak rationale: {rationale}"
        "\nPlease identify the weaknesses and hallucinations of the rationale, and give constructive criticism for improving the weak rationale."
    )

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):
        super().__init__(config)

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt_critique
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
    
    def format_reference(self, passages):
        format_reference = ""
        if passages is None:
            return format_reference
            
        for idx, content in enumerate(passages):
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def get_string(self, query=None, text=None, rationale=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    return self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        input_params = {"query": query, "text": text, "rationale": rationale}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        # print("is_chat", self.is_chat)
        # print("self.enable_chat", self.enable_chat)
        # print("self.is_openai", self.is_openai)
        self.is_chat = True
        if self.is_chat and self.enable_chat:
            # # print("is_chat", self.is_chat)
            # input = []
            # if system_prompt != "":
            #     input.append({"role": "system", "content": system_prompt})
            # if user_prompt != "":
            #     input.append({"role": "user", "content": user_prompt})
            # if not self.is_openai:
            #     input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)
    

class RefinePromptTemplate(BasePromptTemplate):
    placeholders = ["query", "gold", "text", "critique"]

    base_system_prompt_critique = (
        "Read the following documents relevant to the given question: {query}"
        "{text}"
    )
    base_user_prompt = (
        "\nHere is the given weak rationale: {rationale}"
        "\n\nHere is the given critique: {critique}"
        "\nPlease correct the weak rationale based on the critique, and write a better rationale to explain how the contents lead to the answer."
    )

    def __init__(self, config, system_prompt="", user_prompt="", reference_template=None, enable_chat=True):
        super().__init__(config)

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt_critique
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat
        self.reference_template = reference_template
    
    def format_reference(self, passages):
        format_reference = ""
        if passages is None:
            return format_reference
            
        for idx, content in enumerate(passages):
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            if self.reference_template is not None:
                format_reference += self.reference_template.format(idx=idx, title=title, text=text)
            else:
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def get_string(self, query=None, text=None, rationale=None, critique=None, messages=None, **params):
        if messages is not None:
            if isinstance(messages, str):
                return self.truncate_prompt(messages)
            if self.is_chat and self.enable_chat:
                if self.is_openai:
                    return self.truncate_prompt(messages)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    return self.truncate_prompt(prompt)
            else:
                prompt = "\n\n".join(
                    [message['content'] for message in messages if message['content']]
                )
                return self.truncate_prompt(prompt)

        input_params = {"query": query, "text": text, "rationale": rationale, "critique": critique}
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        # print("is_chat", self.is_chat)
        # print("self.enable_chat", self.enable_chat)
        # print("self.is_openai", self.is_openai)
        self.is_chat = True
        if self.is_chat and self.enable_chat:
            # print("is_chat", self.is_chat)
            # input = []
            # if system_prompt != "":
            #     input.append({"role": "system", "content": system_prompt})
            # if user_prompt != "":
            #     input.append({"role": "user", "content": user_prompt})
            # if not self.is_openai:
            #     input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)
