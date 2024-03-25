import streamlit as st

def main():
    show_uploader = True
    
    st.title("File Uploader Example")
    
    if show_uploader:
        uploaded_file = st.file_uploader("Upload a file")
        if uploaded_file:
            show_uploader = False  # Hide the uploader once a file is uploaded
            st.rerun()
    
    if not show_uploader:
        st.write("File uploaded successfully!")
        # Add additional processing logic here if needed

if __name__ == "__main__":
    main()