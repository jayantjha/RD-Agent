wget https://github.com/SunsetWolf/rdagent_resource/releases/download/kaggle_data/kaggle_data.zip
rm -rf git_ignore_folder
mkdir git_ignore_folder
unzip kaggle_data.zip -d git_ignore_folder/kaggle_data
dotenv set KG_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/kaggle_data"
source activate kaggle
dotenv run -- python -m rdagent.app.data_science.loop --competition sf-crime