# Language Chain Project 📚🤖

Welcome to the LangChain Project, a robust tool developed using the langchain package, designed for the analysis and extraction of information from medical literature, with a focus on chest pneumonia. As the final project for the ARTI 502 (Deep Learning) course, the LangChain Project leverages advanced techniques learned in the course to streamline the interpretation of complex medical texts. This initiative stands at the intersection of deep learning, the capabilities of langchain package, and medical research, offering a reliable resource for researchers, medical professionals, and enthusiasts in understanding and combating chest pneumonia.

<br/>
<center>
<img src="images/retrival_qa.png"/>
</center>
<br/>

## Prerequisites 🚀

Ensure you have the required packages installed by running:

```bash
pip install -r requirements.txt
```

## Installation and Setup 🛠️

1. **Load Environment Variables from .env 📥**

- `OPENAI_API_KEY`: Your OpenAI API key. You can get one [here](https://beta.openai.com/).
- ```python
   from dotenv import load_dotenv

   load_dotenv()
   ```

1. **Load Documents 📚**

   ```python
   from langchain.document_loaders import PyPDFLoader

   documents = PyPDFLoader('documents/book_01.pdf').load()
   ```

2. **Preprocess Documents & Create Chunks 📝**

   ```python
   from langchain.text_splitter import CharacterTextSplitter

   chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(documents)
   ```

3. **Create Embeddings 📐**

   ```python
   from langchain.embeddings.openai import OpenAIEmbeddings

   embeddings = OpenAIEmbeddings()
   ```

4. **Initialize The Database 🗄️**

   ```python
   from langchain.vectorstores import Chroma

   database = Chroma.from_documents(chunks, embeddings)
   ```

5. **Create The Retriever 🧲**

   ```python
   retriever = database.as_retriever(search_type='similarity', search_kwargs={'k': 3})
   ```

6. **Import The LLM Model 🤖**

   ```python
   from langchain.llms.openai import OpenAI

   llm = OpenAI()
   ```

7. **Build Your Chain 📜**

   ```python
   from langchain.chains import RetrievalQA

   qa = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type='stuff',
       retriever=retriever,
       return_source_documents=True,
   )
   ```

## Usage 🧪

Now, you're ready to test the system! Use the following functions to ask questions related to chest pneumonia:

```python
ask('What are the symptoms of pneumonia?')
ask('How to treat pneumonia?')
ask('What is the most common cause of pneumonia?')
ask('What is the fatality rate of pneumonia?')
ask('What is the rate of tuberculosis?')
```


## Contributors (Group 5) 👥

- Majid Saleh Al-Raimi
- Rashid Sami Al-Binali
- Mashari Adel Al-Jiban
- **Alwaleed Ahmad Al-Qurashi**
- **Mahmoud Sahal Noor**
- **Abdulrahman Sami Al-Juhani**

## Hardware Specifications 💻
- **Processor:** Apple M2 chip
- **Memory:** 16 GB RAM
- **Storage:** 512 GB SSD
- **Operating System:** macOS Sonoma 14.1.1

## License 📝

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE) file for details.


Feel free to explore and customize the questions to suit your needs. Happy coding! 🚀🤓
