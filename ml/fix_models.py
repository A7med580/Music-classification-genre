import zipfile
import json
import os
import shutil

def fix_keras_model(model_path):
    print(f"Fixing model: {model_path}")
    temp_dir = model_path + "_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Extract
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    config_path = os.path.join(temp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recursively remove quantization_config
        def remove_quant(obj):
            if isinstance(obj, dict):
                if 'quantization_config' in obj:
                    del obj['quantization_config']
                for k, v in obj.items():
                    remove_quant(v)
            elif isinstance(obj, list):
                for item in obj:
                    remove_quant(item)
        
        remove_quant(config)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Re-zip
        os.remove(model_path)
        with zipfile.ZipFile(model_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    zip_ref.write(full_path, rel_path)
        
        print(f"Successfully fixed {model_path}")
    else:
        print(f"No config.json found in {model_path}")
    
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    models_dir = "ml/models"
    for f in os.listdir(models_dir):
        if f.endswith(".keras"):
            fix_keras_model(os.path.join(models_dir, f))
