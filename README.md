# An Image to Speech Application using Huggingface Models:
This application consists of:
## An image to text converter
    The image to text conversion is done using the blip-image-captioning-base
    model made available by Salesforce on the Huggingface Hub.
## Story generator using OpenAI LLM
    A story less than 20 characters is generated using the OpenAI LLM.
## A text to audio generator
    The story is now coverter to audio using the fastspeech2-en-ljspeech model
    from facebook available on the Huggingface Hub.