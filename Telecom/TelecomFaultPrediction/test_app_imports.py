import ast
import os

def test_app_imports_predict():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
        
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
                
    assert "predict" in imported_names, f"'predict' is missing from app.py imports! Found: {imported_names}"
    print("[PASS] 'predict' is properly imported in app.py!")

if __name__ == '__main__':
    test_app_imports_predict()
