from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


# img2text


def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text


# LLM
def generate_story(scenario):
    template = """ 
    You are a story teller: You can generate a short story
    based on a simple narrative. The story should be NOT more than 20 words.
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=['scenario'])
    story_llm = LLMChain(llm=OpenAI(
        model_name='gpt-3.5-turbo', temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


# text2speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        'inputs': message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    print(response.content)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


# scenario = img2text("flower.jpg")
# story = generate_story(scenario)
# text2speech(story)


def main():
    st.set_page_config(page_title='Image to audio story', page_icon='🍎')
    st.header("Turn image into audio story")

    uploaded_file = st.file_uploader('Choose an image...', type='jpg')

    if uploaded_file is not None:
        # Save the uploaded file
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        # Show the uploaded file
        st.image(uploaded_file, caption='Uploaded Image',
                 use_column_width=True)
        # img2text
        scenario = img2text(uploaded_file.name)
        # generate story
        story = generate_story(scenario)
        # text2speech
        text2speech(story)

        with st.expander('scenario'):
            st.write(scenario)
        with st.expander('story'):
            st.write(story)
        st.audio('audio.flac')


if __name__ == '__main__':
    main()
