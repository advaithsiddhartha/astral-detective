import streamlit as st
import nbformat
from nbconvert import HTMLExporter

def display_notebook(notebook_file):
    # Read the notebook file
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook_content = f.read()
    
    # Convert the notebook to HTML
    notebook = nbformat.reads(notebook_content, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    
    # Display the HTML content in Streamlit
    st.components.v1.html(body, height=1200, scrolling=True)

# Streamlit app title
st.title('ASTRAL DETECTIVE')
st.write('IDENTIFYING SPACE CELESTIAL BODIES WITH THE HELP OF MACHINE LEARNING')
st.info('-Advaith Siddhartha')
st.success("About Me .. CLICK HERE =>  : [My Portfolio ](https://advaithsid.web.app/)")
image = st.image("k.png", caption="Sample Image", use_column_width=True)
st.title('THEORY')
st.write('-To be written')
st.title('SOURCE CODE')





# File uploader to upload the .ipynb file
display_notebook('./SPACE.ipynb')
