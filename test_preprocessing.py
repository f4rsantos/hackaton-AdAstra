#!/usr/bin/env python3
"""
Test script to verify image preprocessing pipeline
Tests resizing and chunking logic
"""

from PIL import Image
import numpy as np
from image_preprocessing_pipeline import ImagePreprocessor, preprocess_images_batch

def test_resize_and_chunk():
    """Test the resize and chunking functionality"""
    
    print("=" * 60)
    print("Testing Image Preprocessing Pipeline")
    print("=" * 60)
    
    # Create a test image with the dimensions mentioned: 101708x1229
    print("\n1. Creating test image (101708 x 1229)...")
    test_width = 101708
    test_height = 1229
    
    # Create a simple gradient image for testing
    test_array = np.zeros((test_height, test_width, 3), dtype=np.uint8)
    # Add some gradient pattern
    for i in range(test_height):
        test_array[i, :, :] = int((i / test_height) * 255)
    
    test_image = Image.fromarray(test_array)
    print(f"   ✅ Created test image: {test_image.size}")
    
    # Initialize preprocessor
    print("\n2. Initializing preprocessor...")
    print(f"   Target size: 31778 x 384")
    print(f"   Chunk size: 2048 x 384")
    
    preprocessor = ImagePreprocessor(
        target_width=31778,
        target_height=384,
        chunk_width=2048,
        chunk_height=384
    )
    
    # Test resizing
    print("\n3. Testing resize...")
    resized = preprocessor.resize_image(test_image)
    print(f"   ✅ Resized to: {resized.size}")
    assert resized.size == (31778, 384), f"Resize failed! Expected (31778, 384), got {resized.size}"
    
    # Test chunking
    print("\n4. Testing chunking...")
    chunks = preprocessor.chunk_image(resized)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # Verify chunk properties
    print("\n5. Verifying chunk properties...")
    full_chunks = 0
    partial_chunks = 0
    
    for i, chunk in enumerate(chunks):
        chunk_img = chunk['chunk']
        chunk_width = chunk['width']
        chunk_height = chunk['height']
        x_offset = chunk['x_offset']
        
        print(f"   Chunk {i}: {chunk_width}x{chunk_height} at x={x_offset}")
        
        if chunk_width == 2048:
            full_chunks += 1
        else:
            partial_chunks += 1
        
        # Verify chunk dimensions match metadata
        assert chunk_img.size == (chunk_width, chunk_height), \
            f"Chunk {i} size mismatch! Expected {(chunk_width, chunk_height)}, got {chunk_img.size}"
    
    print(f"\n   ✅ Full-size chunks (2048x384): {full_chunks}")
    print(f"   ✅ Partial chunks: {partial_chunks}")
    
    # Calculate expected number of chunks
    expected_full_chunks = 31778 // 2048
    expected_partial = 1 if 31778 % 2048 > 0 else 0
    expected_total = expected_full_chunks + expected_partial
    
    print(f"\n   Expected: {expected_full_chunks} full + {expected_partial} partial = {expected_total} total")
    assert len(chunks) == expected_total, f"Chunk count mismatch! Expected {expected_total}, got {len(chunks)}"
    
    # Test chunk info
    print("\n6. Testing chunk info...")
    chunk_info = preprocessor.get_chunk_info(chunks)
    print(f"   Total chunks: {chunk_info['total_chunks']}")
    print(f"   Full-size chunks: {chunk_info['full_size_chunks']}")
    print(f"   Partial chunks: {chunk_info['partial_chunks']}")
    print(f"   Last chunk width: {chunk_info['last_chunk_width']}")
    print(f"   Total reconstructed width: {chunk_info['total_width']}")
    
    # Verify last chunk
    last_chunk = chunks[-1]
    expected_last_width = 31778 - (expected_full_chunks * 2048)
    if expected_last_width > 0:
        print(f"\n   ✅ Last chunk is partial: {last_chunk['width']}x{last_chunk['height']}")
        assert last_chunk['width'] == expected_last_width, \
            f"Last chunk width mismatch! Expected {expected_last_width}, got {last_chunk['width']}"
    else:
        print(f"\n   ✅ All chunks are full-size")
    
    # Test full pipeline
    print("\n7. Testing full preprocessing pipeline...")
    resized_image, chunks = preprocessor.process_image(test_image)
    print(f"   ✅ Pipeline complete: {len(chunks)} chunks from {resized_image.size} image")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
    return True

def test_edge_cases():
    """Test edge cases"""
    
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    # Test 1: Image smaller than chunk size
    print("\n1. Testing small image (1024 x 384)...")
    small_image = Image.new('RGB', (1024, 384), color=(128, 128, 128))
    
    preprocessor = ImagePreprocessor(target_width=1024, target_height=384, chunk_width=2048, chunk_height=384)
    resized, chunks = preprocessor.process_image(small_image)
    
    print(f"   ✅ Created {len(chunks)} chunk(s)")
    assert len(chunks) == 1, f"Expected 1 chunk for small image, got {len(chunks)}"
    assert chunks[0]['width'] == 1024, f"Expected chunk width 1024, got {chunks[0]['width']}"
    
    # Test 2: Exact multiple of chunk size
    print("\n2. Testing exact multiple (4096 x 384 with 2048 chunks)...")
    exact_image = Image.new('RGB', (4096, 384), color=(128, 128, 128))
    
    preprocessor = ImagePreprocessor(target_width=4096, target_height=384, chunk_width=2048, chunk_height=384)
    resized, chunks = preprocessor.process_image(exact_image)
    
    print(f"   ✅ Created {len(chunks)} chunks")
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert all(c['width'] == 2048 for c in chunks), "Expected all chunks to be 2048 wide"
    
    # Test 3: One pixel remainder
    print("\n3. Testing one-pixel remainder (2049 x 384)...")
    odd_image = Image.new('RGB', (2049, 384), color=(128, 128, 128))
    
    preprocessor = ImagePreprocessor(target_width=2049, target_height=384, chunk_width=2048, chunk_height=384)
    resized, chunks = preprocessor.process_image(odd_image)
    
    print(f"   ✅ Created {len(chunks)} chunks")
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert chunks[0]['width'] == 2048, f"Expected first chunk 2048, got {chunks[0]['width']}"
    assert chunks[1]['width'] == 1, f"Expected last chunk 1, got {chunks[1]['width']}"
    
    print("\n" + "=" * 60)
    print("✅ ALL EDGE CASE TESTS PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        # Run main test
        test_resize_and_chunk()
        
        # Run edge case tests
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe preprocessing pipeline is working correctly:")
        print("✅ Images resize properly from 101708x1229 to 31778x384")
        print("✅ Images chunk properly into 2048x384 pieces")
        print("✅ Remaining small chunks are handled correctly")
        print("✅ All edge cases pass")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
