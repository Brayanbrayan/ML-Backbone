import asyncio
from googletrans import Translator

async def translate_text():
    # 1. Define the text
    text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
    
    # 2. Create a Translator object
    translator = Translator()

    # 3. Translate the text using 'await'
    # The 'await' keyword tells Python to pause here until the translation is complete.
    translation = await translator.translate(text, dest='fr')

    # 4. Print the translated text
    print(translation.text)

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(translate_text())