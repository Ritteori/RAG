
from app.services.retrieval import group_by_files, \
    find_neighbours, \
    build_context_texts, \
    find_anchor_chunks_scores, \
    find_top_k_contexts

def test_group_by_files(chunked_texts):
    searches = {
        0: [
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::0",
                "score": 0.9,
                "category": "math"
            },
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::1",
                "score": 0.8,
                "category": "math"
            }
        ]
    }

    grouped = group_by_files(searches, chunked_texts)

    assert len(grouped) == 1
    assert grouped["/data/math/vectors_matrix_definition.txt"][0]["chunk_index"] == 0

def test_find_neighbours(chunked_texts):
    grouped = {
        '/data/math/vectors_matrix_definition.txt':[
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::0",
                "chunk_index": 0,
                "score": 0.9,
            },
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::1",
                "chunk_index": 1,
                "score": 0.8,
            }
        ]
    }

    neighbours = find_neighbours(grouped, chunked_texts)

    assert neighbours['/data/math/vectors_matrix_definition.txt']["anchor_chunks"] == [0,1]
    assert set(neighbours['/data/math/vectors_matrix_definition.txt']["context_chunks"]) == {0,1,2}
    

def test_build_context_texts(chunked_texts):
    
    neighbours = {
        '/data/math/vectors_matrix_definition.txt':{
                'anchor_chunks':[0,1],
                'context_chunks':[0,1,2]
        }
    }

    contexts = build_context_texts(neighbours, chunked_texts)
    
    assert isinstance(contexts[0],str)

def test_find_anchor_chunks_scores():

    searches = {
        0: [
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::0",
                "score": 0.9,
                "category": "math"
            },
            {
                "chunk_id": "data/math/vectors_matrix_definition.txt::1",
                "score": 0.8,
                "category": "math"
            }
        ]
    }

    neighbours = {
        '/data/math/vectors_matrix_definition.txt':{
                'anchor_chunks':[0,1],
                'context_chunks':[0,1,2]
        }
    }
    
    chunk_score, best_contexts = find_anchor_chunks_scores(searches, neighbours)

    assert chunk_score["data/math/vectors_matrix_definition.txt::0"] == 0.9
    assert best_contexts == [0.9, 0.8]


def test_find_top_k_contexts():
    
    contexts = ['asdasda', 'adasdasd']
    best_contexts = [0.9,0.8]

    top_k_contexts = find_top_k_contexts(contexts,best_contexts,2)

    assert top_k_contexts[0][1] == 0.9
    assert len(top_k_contexts) == 2