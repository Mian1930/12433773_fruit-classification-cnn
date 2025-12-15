# tests/test_preprocessing.py
import pytest
from data_loader import create_generators

def test_generators_shapes(tmp_path):
    # This test assumes you put a tiny sample dataset under tmp_path
    # Create a tiny structure: tmp_path/classA/img1.jpg ...
    import os
    from PIL import Image
    classA = tmp_path / "apple"
    classB = tmp_path / "banana"
    classA.mkdir()
    classB.mkdir()
    # Create tiny images
    for i in range(3):
        Image.new('RGB',(100,100)).save(classA / f"img{i}.jpg")
        Image.new('RGB',(100,100)).save(classB / f"img{i}.jpg")

    train_gen, val_gen, test_gen, class_indices = create_generators(str(tmp_path), batch_size=2, val_split=0.2, test_split=0.2)
    # Check class indices
    assert 'apple' in class_indices and 'banana' in class_indices
    # Check generator yields images of expected shape
    x,y = next(train_gen)
    assert x.shape[1:] == (128,128,3)
