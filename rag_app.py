import streamlit as st #all streamlit commands will be available through the "st" alias
import rag_lib as glib #reference to local lib script


st.set_page_config(page_title="Retrieval-Augmented Generation") #HTML title
st.title("Team Teacher:\nCurriculum specific personalised answers") #page title



if 'vector_index' not in st.session_state: #see if the vector index hasn't been created yet
    with st.spinner("Indexing document..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index() #retri eve the index through the supporting library and store in the app's session cache

options = ["Isabella", "Oliver", "Lucas", "Mia"] #list of options for the persona selector

STUDENT_INFO = {"Isabella": """ - Background: Isabella has a shy and introverted personality, making it hard for her to make friends.
   - School Performance: Below average in most subjects, particularly in Oral Presentations; excels in Art.
   - Interests: Drawing, Anime, Fantasy Novels, Piano.
   - Skill Level: Very talented in drawing; uses art as a form of expression.
""",
                "Oliver": """ - Background: Oliver's family has faced financial difficulties, impacting his access to resources and extracurricular activities.
   - School Performance: Struggles academically, especially in advanced subjects like Mathematics and Science.
   - Interests: Hip-hop Music, DIY Projects, Thrift Shopping, Cooking.
   - Skill Level: Good at practical tasks and DIY projects; cooking is a necessity turned into an interest.
""",
                "Lucas": """  - Background: Lucas lives with his single mother who works two jobs. He often has to care for his younger siblings.
   - School Performance: Inconsistent performance across subjects; often tired or unprepared for school.
   - Interests: Comic Books, Action Movies, Soccer, Volunteer Work.
   - Skill Level: Enthusiastic about soccer but rarely gets the chance to play organised sports.
""",
                "Mia": """   - Background: Mia's family recently moved to Sydney from a rural area, and she finds it hard to adjust to the city life.
   - School Performance: Falling behind in several subjects, especially in Science and Social Studies; language barrier issues.
   - Interests: Country Music, Reading, Gardening, Baking.
   - Skill Level: Enjoys reading but struggles with comprehension; talented in baking.
"""}


student_selector = st.selectbox("Persona", options, index=0, placeholder="Choose a Student")

st.write("Student: ", student_selector)
st.write("Student Info: \n", STUDENT_INFO[student_selector])

input_text = st.text_area("Input text", label_visibility="collapsed") #display a multiline text box with no label
go_button = st.button("Go", type="primary") #display a primary button

if go_button: #code in this if block will be run when the button is clicked
    with st.spinner("Working..."): #show a spinner while the code in this with block runs
        response_content = glib.get_rag_response(index=st.session_state.vector_index, question=input_text) #call the model through the supporting library
        st.write(response_content)
        personalised_content = glib.get_custom_response(temp_answer=response_content, student_info=STUDENT_INFO[student_selector])
        st.write(personalised_content)
        # response_content = glib.get_rag_response2(question=input_text)
        # st.write(response_content) #display the response content
        
