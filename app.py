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
    st.components.v1.html(body, height=1800, scrolling=True)

# Streamlit app title
st.title('ASTRAL DETECTIVE')
st.write('IDENTIFYING SPACE CELESTIAL BODIES WITH THE HELP OF MACHINE LEARNING')
st.info('-Advaith Siddhartha')
st.write("Data set used  =>  : [DATA SET ](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey)")
st.success("About Me .. CLICK HERE =>  : [My Portfolio ](https://advaithsid.web.app/)")
image = st.image("k.png", caption="Image Processing CLR", use_column_width=True)
st.title('THEORY')
st.write('''Hey There !
My machine learning project revolves around leveraging the power of K-Nearest Neighbors (KNN) and Linear Regression to identify celestial objects such as stars, galaxies, and planets. Utilizing the rich and expansive Sloan Digital Sky Survey (SDSS) DR14 dataset from Kaggle, this project aims to classify these astronomical entities accurately. The SDSS DR14 dataset is renowned for its comprehensive collection of data points, which include detailed observations of celestial bodies, encompassing attributes such as their magnitude in various filters (u, g, r, i, z), as well as redshift values. This data provides a fertile ground for training machine learning models.

The initial phase of the project involved extensive data preprocessing. This included handling missing values, normalizing the data, and splitting it into training and test sets. Given the multidimensional nature of the dataset, dimensionality reduction techniques like Principal Component Analysis (PCA) were also explored to enhance model performance and reduce computational overhead.

The first model employed was the K-Nearest Neighbors algorithm. KNN is particularly suited for classification problems due to its simplicity and effectiveness in capturing complex patterns. The model was trained to classify objects based on their photometric data. Hyperparameter tuning was performed to determine the optimal value of K, ensuring that the model achieved a balance between bias and variance. Cross-validation was employed to validate the model's performance and mitigate overfitting.

In parallel, Linear Regression was utilized to predict continuous variables such as redshift, which is crucial for understanding the distance and velocity of celestial bodies. The linear regression model was trained using the least squares method, and regularization techniques like Ridge and Lasso regression were explored to prevent overfitting and enhance model generalization. The model's performance was evaluated using metrics such as Mean Squared Error (MSE) and R-squared (R²).

Despite the promising results, a significant challenge emerged: the scarcity of labeled images for training a more sophisticated model capable of analyzing astronomical pictures directly. This limitation prompted further modifications to the project. One potential enhancement involves leveraging transfer learning, a powerful technique in deep learning. By using pre-trained models like Convolutional Neural Networks (CNNs) that have been trained on large datasets (such as ImageNet), the project can benefit from the knowledge these models have already acquired. Fine-tuning these pre-trained models with the available astronomical images could significantly improve their classification accuracy, even with a limited dataset.

Another avenue for improvement is the use of data augmentation techniques. By artificially increasing the size of the training set through transformations such as rotation, scaling, and flipping of existing images, the model can be made more robust and better equipped to generalize to unseen data. Additionally, synthetic data generation methods, such as Generative Adversarial Networks (GANs), could be explored to create realistic images of celestial bodies, further enriching the training dataset.

To enhance the project’s applicability and performance, integrating unsupervised learning techniques such as clustering (e.g., K-means or DBSCAN) could help identify underlying patterns in the data that might not be immediately apparent. These techniques could uncover new insights into the classification of celestial objects and potentially lead to the discovery of novel astronomical phenomena.

Overall, this machine learning project on identifying stars, galaxies, and planets from the SDSS DR14 dataset not only demonstrates the utility of KNN and Linear Regression but also highlights the challenges and potential solutions in dealing with limited image data. The planned enhancements, including transfer learning, data augmentation, and synthetic data generation, aim to overcome these challenges, paving the way for more accurate and comprehensive astronomical object classification.''')
st.title('SOURCE CODE phase - 1')





# File uploader to upload the .ipynb file
display_notebook('SPACE.ipynb')
