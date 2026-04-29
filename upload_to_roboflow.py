from dotenv import load_dotenv
import roboflow
import os

load_dotenv()

# Temporary debug
api_key = os.getenv("ROBOFLOW_API_KEY")


rf = roboflow.Roboflow(api_key=api_key)
project = rf.workspace("dogukans-workspace-6uczq").project("toronto-road-intel")
project.upload("output/all_frames/20260427")
