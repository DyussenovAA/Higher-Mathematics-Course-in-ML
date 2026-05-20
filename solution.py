import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    # Чтение входных фраз
    with open('input.txt', 'r', encoding='utf-8') as f:
        passphrases = [line.strip() for line in f if line.strip()]

    # Загрузка модели и токенизатора
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Пайплайн для генерации
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    passwords = []
    for phrase in passphrases:
        # Формируем сообщение и применяем шаблон чата
        messages = [
            {"role": "user", "content": f"Just generate a continuation of the phrase: {phrase}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Генерация с ограничением на 16 новых токенов
        # Используем жадный декодинг для воспроизводимости
        outputs = gen_pipeline(
            prompt,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            return_full_text=False
        )
        generated_text = outputs[0]['generated_text']

        # Ищем первую подстроку из 16 символов [a-zA-Z0-9]
        match = re.search(r'[a-zA-Z0-9]{16}', generated_text)
        if match:
            password = match.group(0)
        else:
            # fallback: берём первые 16 символов после очистки от пробелов
            clean = re.sub(r'\s+', '', generated_text)
            password = clean[:16]
        passwords.append(password)

    # Сохранение результата
    with open('output.txt', 'w', encoding='utf-8') as f:
        for pwd in passwords:
            f.write(pwd + '\n')

if __name__ == '__main__':
    main()