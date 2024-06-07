import streamlit as st
import pandas as pd
import subprocess
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType, DoubleType, StringType, ArrayType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
import traceback
import os

st.title("IDE | PySpark")
st.write("Muhammad Ammar Hassan")
st.write("SP21-BDS-014")
st.sidebar.title("Options")

# Initialize SparkSession
try:
    from pyspark.sql import SparkSession
    SparkSession.builder.getOrCreate().stop()
except Exception as e:
    st.sidebar.warning("No existing Spark session found to stop.")

try:
    spark = SparkSession.builder \
        .appName("Streamlit IDE") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    #st.sidebar.success("Spark Session initialized successfully")
except Exception as e:
    st.sidebar.error(f"Error initializing Spark session: {e}")
    st.sidebar.error(traceback.format_exc())

# # File uploader
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a dataset", type=['csv', 'xlsx', 'parquet', 'json', 'orc', 'txt'])

if uploaded_file and 'spark' in locals():
    try:
        # Save uploaded file temporarily to be read by Spark
        with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Determine file type and read into Spark DataFrame
        if uploaded_file.name.endswith('.csv'):
            sdf = spark.read.csv(f"/tmp/{uploaded_file.name}", header=True, inferSchema=True)
        elif uploaded_file.name.endswith('.parquet'):
            sdf = spark.read.parquet(f"/tmp/{uploaded_file.name}")
        elif uploaded_file.name.endswith('.json'):
            sdf = spark.read.json(f"/tmp/{uploaded_file.name}")
        elif uploaded_file.name.endswith('.orc'):
            sdf = spark.read.orc(f"/tmp/{uploaded_file.name}")
        elif uploaded_file.name.endswith('.txt'):
            sdf = spark.read.csv(f"/tmp/{uploaded_file.name}", sep='\t', header=True, inferSchema=True)
        
        st.subheader("Dataset")
        # Print SDF      
        st.write(sdf.limit(10).toPandas())
    
        # # Preprocessing options
        st.sidebar.header("Data Preprocessing")
        if st.sidebar.checkbox("Dataset summary"):
            summary = sdf.describe()
            st.subheader("Summary")
            st.write(summary)

        if st.sidebar.checkbox("Remove Duplicates"):
            sdf = sdf.dropDuplicates()

        numeric_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))]

        if st.sidebar.checkbox("Remove outliers"):
            for col_name in numeric_cols:
                quantiles = sdf.approxQuantile(col_name, [0.25, 0.75], 0.05)
                Q1, Q3 = quantiles[0], quantiles[1]
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                sdf = sdf.filter((F.col(col_name) >= lower_bound) & (F.col(col_name) <= upper_bound))

        if st.sidebar.checkbox("Encode categorical variable"):
            categorical_cols = [field.name for field in sdf.schema.fields if isinstance(field.dataType, StringType)]

            for col_name in categorical_cols:
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
                sdf = indexer.fit(sdf).transform(sdf)
            
                # Check if the indexed column has at least two distinct values
                distinct_count = sdf.select(f"{col_name}_indexed").distinct().count()
                if distinct_count > 1:
                    encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_encoded")
                    sdf = encoder.fit(sdf).transform(sdf)
                    # Remove the original and indexed columns
                    sdf = sdf.drop(col_name, f"{col_name}_indexed")
                else:
                    # Rename the indexed column to encoded column
                    sdf = sdf.withColumnRenamed(f"{col_name}_indexed", f"{col_name}_encoded")
                    # Remove the original column
                    sdf = sdf.drop(col_name)

        if st.sidebar.checkbox("Handle Null Values"):
            strategy = st.sidebar.selectbox("Choose a strategy", ["Select Anyone from the list","Drop", "Fill with constant", "Fill with Mean", "Fill with Median", "Fill with Mode"])
            if strategy == "Select Anyone from the list":
                pass
            elif strategy == "Drop":
                sdf = sdf.dropna()
            elif strategy == "Fill with constant":
                # Find columns with missing values
                cols_with_missing = [col for col in numeric_cols if sdf.filter(F.col(col).isNull()).count() > 0]
    
                if not cols_with_missing:
                    st.warning("No missing values in the dataset.")
                else:
                    constant_values = {}
                    for col in cols_with_missing:
                        fill_value = st.sidebar.text_input(f"Enter constant value for {col}", key=col)
                        if fill_value:
                            constant_values[col] = float(fill_value) if '.' in fill_value else int(fill_value)

                    if not constant_values:
                        st.warning("Please enter values to fill.")
                    else:
                        sdf = sdf.na.fill(constant_values)

            elif strategy == "Fill with Mean":
                mean_values = {c: sdf.select(F.mean(F.col(c)).alias(c)).first()[c] for c in numeric_cols}
                sdf = sdf.na.fill(mean_values)
            elif strategy == "Fill with Median":
                median_values = {c: sdf.select(F.expr(f'percentile_approx({c}, 0.5)').alias(c)).first()[c] for c in numeric_cols}
                sdf = sdf.na.fill(median_values)
            elif strategy == "Fill with Mode":
                mode_values = {c: sdf.groupBy(c).count().orderBy('count', ascending=False).first()[0] for c in numeric_cols}
                sdf = sdf.na.fill(mode_values)

        st.subheader("Processed Dataset")
        st.write(sdf.limit(10).toPandas())  # Display first 10 rows for performance

        # Model selection
        st.sidebar.header("Model Selection")
        model_type = st.sidebar.selectbox("Select Model Type", ["Classification", "Clustering"])
        features = st.sidebar.multiselect("Select features", sdf.columns)

        split_ratio = st.sidebar.slider("Select test, train split size", 0.1, 0.9, 0.7)
        train_data, test_data = sdf.randomSplit([split_ratio, 1 - split_ratio])

        assembler = VectorAssembler(inputCols=features, outputCol="features")
        train_data = assembler.transform(train_data)
        test_data = assembler.transform(test_data)

        if model_type == "Classification":
            target = st.sidebar.selectbox("Select target", sdf.columns)
            classifier = st.sidebar.selectbox("Select Classifier", ["Decision Tree", "Logistic Regression", "Random Forest", "Naive Bayes", "SVM", "Linear Regression"])
            if classifier == "Decision Tree":
                model = DecisionTreeClassifier(labelCol=target, featuresCol="features")
            elif classifier == "Logistic Regression":
                model = LogisticRegression(labelCol=target, featuresCol="features")
            elif classifier == "Random Forest":
                model = RandomForestClassifier(labelCol=target, featuresCol="features")
            elif classifier == "Naive Bayes":
                model = NaiveBayes(labelCol=target, featuresCol="features")
            elif classifier == "SVM":
                model = LinearSVC(labelCol=target, featuresCol="features")
            elif classifier == "Linear Regression":
                model = LinearRegression(labelCol=target, featuresCol="features")

        elif model_type == "Clustering":
            clusterer = st.sidebar.selectbox("Select Clusterer", ["K-Means", "Gaussian Mixture Model", "Hierarchical Clustering"])
            if clusterer == "K-Means":
                num_clusters = st.sidebar.slider("Select number of clusters", 2, 10)
                model = KMeans(featuresCol="features", k=num_clusters)
            elif clusterer == "Gaussian Mixture Model":
                num_clusters = st.sidebar.slider("Select number of clusters", 2, 10)
                model = GaussianMixture(featuresCol="features", k=num_clusters)
            elif clusterer == "Hierarchical Clustering":
                num_clusters = st.sidebar.slider("Select number of clusters", 2, 10)
                model = BisectingKMeans(featuresCol="features", k=num_clusters)        

        if st.sidebar.button("Train Model"):
            with st.spinner('Training the model...'):
                try:
                    model = model.fit(train_data)
                
                    if model_type == "Classification":
                        predictions = model.transform(test_data)
                        evaluator = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")
                        accuracy = evaluator.evaluate(predictions)
                        st.subheader("Model Accuracy")
                        st.write(f"Accuracy: {accuracy:.2f}")
                        st.subheader("Predictions")
                        st.write(predictions.select("features", target, "prediction").limit(10).toPandas())  # Display first 10 predictions
                
                    elif model_type == "Clustering":
                        predictions = model.transform(sdf)
                        evaluator = ClusteringEvaluator()
                        silhouette = evaluator.evaluate(predictions)
                        st.subheader("Model Evaluation")
                        st.write(f"Silhouette with squared euclidean distance: {silhouette:.2f}")
                        st.subheader("Cluster Centers")
                        if hasattr(model, 'clusterCenters'):
                            st.write(model.clusterCenters())
                        st.subheader("Predictions")
                        st.write(predictions.select("features", "prediction").limit(10).toPandas())  # Display first 10 predictions

                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
                    st.error(traceback.format_exc())

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error(traceback.format_exc())
else:
    st.info("Please upload a dataset to proceed.")
