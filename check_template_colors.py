#!/usr/bin/env python3
"""
Quick script to check if templates are colored or grayscale
"""
import cv2 as cv
import numpy as np
from pathlib import Path

template_dir = Path("stored_templates")
template_files = list(template_dir.rglob("*.png"))

print(f"Found {len(template_files)} template files\n")

for template_file in template_files:
    template = cv.imread(str(template_file), cv.IMREAD_COLOR)
    if template is not None:
        b, g, r = cv.split(template)
        
        is_grayscale = np.array_equal(b, g) and np.array_equal(g, r)
        
        print(f"Template: {template_file.name}")
        print(f"  Shape: {template.shape}")
        print(f"  Is Grayscale: {is_grayscale}")
        print(f"  BGR means: B={np.mean(b):.1f}, G={np.mean(g):.1f}, R={np.mean(r):.1f}")
        
        if is_grayscale:
            print(f"  ⚠️ This template is GRAYSCALE - all channels are identical")
        else:
            print(f"  ✅ This template is COLORED - channels are different")
        print()
