import os
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory



def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        # credentials_profile_name=
        # os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",
        # os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings(
        # credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1" #sets the region name (if not the default)
        # endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
    ) #create a Titan Embeddings client
    
    pdf_path = "Text_Book_for_Year_6_Science_Knowledge.pdf" #assumes local PDF file with this name

    loader = PyPDFLoader(file_path=pdf_path) #load the pdf file
    
    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000, #divide into 1000-character chunks using the separators above
        chunk_overlap=100 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF

    
    return index_from_loader #return the index to be cached by the client app

def get_rag_response2(question): #rag client function
    llm = get_llm()

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    predicted_text = conversation.predict(input="Hi there!")
    
    # response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    
    print(predicted_text)

    return predicted_text

def get_rag_response(index, question): #rag client function
    llm = get_llm()


    # result = index.query_index(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    # print(result)
    response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    print(response_text)

    return response_text

def get_custom_response(temp_answer):
    llm = get_llm()

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    student = """Ethan, 13, Sydney, Australia
   - Background: Ethan comes from a tech-savvy family in the suburbs of Sydney. His parents are software engineers.
   - School Performance: Strong in Mathematics and Science; moderate in Humanities.
   - Interests: Robotics, Coding, Chess, Science Fiction Novels, Astronomy.
   - Skill Level: Advanced in Robotics and Coding; participates in regional competitions.
"""

    new_question = f"Please modify the answer: {temp_answer} so it is personalised to {student}.\n Make a relevant analogy that the student would understand. Begin your answer with 'Think about it like: '"

    personal_answer = conversation.predict(input=new_question)

    return personal_answer



# Prompt them to think more rather than just giving them the answer