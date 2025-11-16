"""Quick test script to verify data loading works correctly"""

import json
import sys
from pathlib import Path

# Add LCoT2Tree to path
sys.path.append(str(Path(__file__).parent.parent / "LCoT2Tree" / "src" / "gnn"))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)

def test_data_structure(data_path: str):
    """Test if the data file has the correct structure"""
    
    print(f"Testing data file: {data_path}")
    print("=" * 60)
    
    if not Path(data_path).exists():
        print(f"❌ Error: File not found: {data_path}")
        return False
    
    print("✓ File exists")
    
    # Try to load and parse the file
    try:
        with open(data_path, "r") as f:
            lines = f.readlines()
        
        print(f"✓ File has {len(lines)} lines")
        
        # Check first line
        if len(lines) == 0:
            print("❌ Error: File is empty")
            return False
        
        # Parse first entry
        first_entry = json.loads(lines[0])
        print("✓ First line is valid JSON")
        
        # Check required fields
        required_fields = ["thoughts_list", "cot_tree", "score"]
        missing_fields = []
        
        for field in required_fields:
            if field not in first_entry:
                missing_fields.append(field)
            else:
                print(f"✓ Field '{field}' present")
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        # Check thoughts_list structure
        thoughts_list = first_entry["thoughts_list"]
        if isinstance(thoughts_list, str):
            thoughts_list = json.loads(thoughts_list)
        
        print(f"✓ thoughts_list has {len(thoughts_list)} thoughts")
        
        # Check cot_tree structure
        cot_tree = first_entry["cot_tree"]
        required_tree_fields = ["value", "level", "cate", "thought_list", "children"]
        
        for field in required_tree_fields:
            if field not in cot_tree:
                print(f"❌ Missing cot_tree field: {field}")
                return False
        
        print(f"✓ cot_tree has correct structure")
        print(f"  - Root value: {cot_tree['value']}")
        print(f"  - Root level: {cot_tree['level']}")
        print(f"  - Number of children: {len(cot_tree['children'])}")
        
        # Check score
        score = first_entry["score"]
        if score not in ["0", "1", 0, 1]:
            print(f"⚠ Warning: Score is '{score}' (expected '0' or '1')")
        else:
            print(f"✓ Score is valid: {score}")
        
        # Count label distribution
        correct_count = 0
        incorrect_count = 0
        
        for line in lines:
            try:
                entry = json.loads(line)
                score = str(entry.get("score", ""))
                if score == "1":
                    correct_count += 1
                elif score == "0":
                    incorrect_count += 1
            except:
                pass
        
        print(f"\n✓ Label distribution:")
        print(f"  - Correct answers: {correct_count} ({100*correct_count/len(lines):.1f}%)")
        print(f"  - Incorrect answers: {incorrect_count} ({100*incorrect_count/len(lines):.1f}%)")
        
        # Test tokenizer
        print(f"\n✓ Testing tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            first_thought = list(thoughts_list.values())[0]
            tokens = tokenizer.encode(first_thought)
            print(f"  - First thought has {len(tokens)} tokens")
            print(f"  - Preview: {first_thought[:100]}...")
        except Exception as e:
            print(f"❌ Error with tokenizer: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✓ All checks passed! Data is ready for training.")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "deepseek/final.json"
    
    success = test_data_structure(data_path)
    
    if success:
        print("\n" + "=" * 60)
        print("Ready to train! Run:")
        print(f"  python scripts/02_train_model.py --data_path {data_path} --save_model")
        print("Or:")
        print(f"  ./scripts/train_gat_simple.sh {data_path}")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n❌ Data validation failed. Please fix the issues above.")
        sys.exit(1)

