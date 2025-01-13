from openai import OpenAI


class OpenAILLM:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def prompt(
        self, prompt, response_prefix='', max_tokens=2048, max_pages=5, stop_sequences=None,
        **kwargs
    ):
        num_pages_remaining = max_pages
        response_pages = []
        if response_prefix:
            # NOTE: OpenAI seems to prefer this over putting it in assistant
            prompt += response_prefix
        # paginate for long responses
        while num_pages_remaining:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            if response_pages:
                prev_response = ''.join(response_pages)
                messages.append({
                    "role": "assistant",
                    "content": prev_response
                })

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages,
                stop=stop_sequences,
                **kwargs
            )
            response_text = response.choices[0].message.content
            response_pages.append(response_text)
            if response.choices[0].finish_reason != 'length':
                break
            else:
                num_pages_remaining -= 1

        return ''.join(response_pages)
