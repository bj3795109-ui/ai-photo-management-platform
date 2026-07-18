from pipeline.embedding import hamming_distance

def find_phash_duplicates(image_hashes: dict, threshold: int = 5) -> list[tuple]:
    duplicates=[]
    paths = list(image_hashes.keys())
    
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            path1, path2 = paths[i], paths[j]
            dist = hamming_distance(image_hashes[path1], image_hashes[path2])
            if dist <= threshold:
                duplicates.append((path1,path2, dist))
                
    return duplicates

import numpy as np

def find_clip_duplicates(embeddings: dict, threshold: float = 0.95) -> list[tuple]:
    duplicates = []
    paths = list(embeddings.keys())
    
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            path1, path2 = paths[i], paths[j]
            similarity = np.dot(embeddings[path1], embeddings[path2])
            if similarity >= threshold:
                duplicates.append((path1, path2,similarity))
    
    return duplicates