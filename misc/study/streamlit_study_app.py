import streamlit as st
import tempfile
from PyPDF2 import PdfReader
import os
from openai import OpenAI

# Set up OpenAI
# openai.api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
client = OpenAI()

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_summary(text):
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful study assistant."},
    #         {"role": "user", "content": f"Create concise bullet point summaries from this text:\n{text}"}
    #     ]
    # )
    # return response.choices[0].message['content']
    print("SUMMARY")
    response = client.chat.completions.create(
        # model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        model="gpt-4.1-nano-2025-04-14",  # or "gpt-3.5-turbo"
        # model="gpt-4.1-nano",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful study assistant."},
            {"role": "user", "content": f"Create concise bullet point summaries from this text:\n{text}"}
        ]
    )
    return response.choices[0].message.content
def generate_quiz(text):
    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        # model="gpt-4.1-nano", #gpt-4.1-nano
        model="gpt-4.1-nano-2025-04-14", #gpt-4.1-nano
        messages=[
            {"role": "system", "content": """Generate 5 multiple choice questions based on the text. 
                Format each question like this:
                
                Question: [question text]
                Options:
                A) [option 1]
                B) [option 2]
                C) [option 3]
                D) [option 4]
                Answer: [correct letter]
                
                Separate questions with '---'"""},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def parse_quiz_response(quiz_text):
    questions = []
    current_question = {}
    
    for line in quiz_text.split('\n'):
        line = line.strip()
        if line.startswith("Question:"):
            current_question['question'] = line.replace("Question:", "").strip()
            current_question['options'] = []
        elif line.startswith(("A)", "B)", "C)", "D)")):
            current_question['options'].append(line)
        elif line.startswith("Answer:"):
            current_question['answer'] = line.replace("Answer:", "").strip()
            questions.append(current_question)
            current_question = {}
    
    return questions

# Main app
st.title("Study Assistant 📚")

uploaded_file = st.file_uploader("Upload PDF or Text file", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode()
    
    st.session_state['document_text'] = text

# Sidebar menu
menu_option = st.sidebar.radio("Select Option", ["Summary", "Quiz"])

if 'document_text' in st.session_state:
    if menu_option == "Summary":
        st.subheader("Document Summary")
        if st.button("Generate Summary"):
            with st.spinner("Creating summary..."):
                summary = generate_summary(st.session_state.document_text)
                st.write(summary)
    
    elif menu_option == "Quiz":
        st.subheader("Quiz")
        if 'quiz_questions' not in st.session_state:
            st.session_state.quiz_questions = []
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
        
        if st.button("Generate Quiz"):
            with st.spinner("Creating quiz questions..."):
                quiz_response = generate_quiz(st.session_state.document_text)
                st.session_state.quiz_questions = parse_quiz_response(quiz_response)
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = {}
        
        if st.session_state.quiz_questions:
            for i, question in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Question {i+1}:** {question['question']}")
                
                # Display options
                key = f"q{i}"
                options = [opt.split(") ", 1)[1] for opt in question['options']]
                st.session_state.user_answers[key] = st.radio(
                    f"Options for Question {i+1}",
                    options,
                    key=key,
                    index=None
                )
            
            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True
                score = 0
                for i, question in enumerate(st.session_state.quiz_questions):
                    user_answer = st.session_state.user_answers.get(f"q{i}")
                    correct_answer = question['options'][ord(question['answer'].lower()) - 97].split(") ", 1)[1]
                    
                    if user_answer == correct_answer:
                        score += 1
                
                st.success(f"Your score: {score}/{len(st.session_state.quiz_questions)}")
                
                # Show answers
                for i, question in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Question {i+1}:** {question['question']}")
                    correct_answer = question['options'][ord(question['answer'].lower()) - 97].split(") ", 1)[1]
                    user_answer = st.session_state.user_answers.get(f"q{i}")
                    
                    if user_answer:
                        if user_answer == correct_answer:
                            st.markdown(f"✅ Your answer: **{user_answer}** (Correct)")
                        else:
                            st.markdown(f"❌ Your answer: **{user_answer}**")
                            st.markdown(f"Correct answer: **{correct_answer}**")
                    else:
                        st.markdown("⚠️ You didn't answer this question")
                        st.markdown(f"Correct answer: **{correct_answer}**")

else:
    # if not openai.api_key:
    #     st.sidebar.error("Please enter your OpenAI API key")
    # else:
    st.info("Please upload a document to get started")