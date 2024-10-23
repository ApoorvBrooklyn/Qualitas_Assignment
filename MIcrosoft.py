from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import numpy as np
import json
import threading
import time
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AzureDocProcessor:
    def extract_text_n_tables(self, doc_path):
        def format_bounding_region(bounding_regions):
            if not bounding_regions:
                return "N/A"
            return ", ".join("Page #{}: {}".format(region.page_number, format_polygon(region.polygon)) for region in bounding_regions)
        def format_polygon(polygon):
            if not polygon:
                return "N/A"
            return "["+", ".join(["[{}, {}]".format(p.x, p.y) for p in polygon])+"]"
        # helper fns---------------------------------------------
        def add_to_dict_of_lists(target_dict, key, item_to_add_to_list): #note dict is passed by reference
            if target_dict.get(key) is None:
                target_dict[key] = [item_to_add_to_list]
            else:
                target_dict[key].append(item_to_add_to_list)
        def ignore_para(para_pgno, para_bbox, bboxes_to_ignore):
            def bbox1_contains_bbox2(bbox1, bbox2): #1 is big, #2 is small
                topleft_1 = bbox1[0]
                topleft_2 = bbox2[0]
                botright_1 = bbox1[2]
                botright_2 = bbox2[2]
                if topleft_1[0] <= topleft_2[0] + 0.1 and topleft_1[1] <= topleft_2[1] + 0.1:
                    if botright_1[0] >= botright_2[0] - 0.1 and botright_1[1] >= botright_2[1] - 0.1:
                        return True
                return False
            if para_pgno not in bboxes_to_ignore.keys():
                return False
            para_bbox = eval(para_bbox)
            for bbox in bboxes_to_ignore[para_pgno]:
                bbox = eval(bbox)
                if bbox1_contains_bbox2(bbox, para_bbox):
                    return True
            return False
        # exec---------------------------------------
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        with open(doc_path, "rb") as f:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-layout", document=f, locale="en-US"
            )
        result = poller.result()
        tables = {}
        contents = {}
        bboxes_to_ignore = {}
        footers = {}
        for table_idx, table in enumerate(result.tables):
            region = table.bounding_regions[0]
            if bboxes_to_ignore.get(region.page_number) is None:
                bboxes_to_ignore[region.page_number] = [format_polygon(region.polygon)]
            else:
                bboxes_to_ignore[region.page_number].append(format_polygon(region.polygon))
            np_table = np.empty((table.row_count, table.column_count)).tolist()
            for i in range(len(np_table)):
                for j in range(len(np_table[0])):
                    np_table[i][j] = ''
            for cell in table.cells:
                np_table[cell.row_index][cell.column_index] = cell.content
            add_to_dict_of_lists(tables, region.page_number, np_table)
        for para in result.paragraphs:
            region = para.bounding_regions[0]
            bbox = format_polygon(region.polygon)
            pgno = region.page_number
            if not ignore_para(pgno, bbox, bboxes_to_ignore):
                add_to_dict_of_lists(contents, pgno, para.content)
        for para in result.paragraphs:
            region = para.bounding_regions[0]
            if para.role == "footnote" or para.role=="pageFooter":
                footers[region.page_number] = para.content

        extracted_data = {
            doc_path: {
                'contents':contents,
                'tables':tables,
                "footers": footers,
            }
        }
        return extracted_data
    def prepare_to_chunks(self, extracted_data):
        chunked_slidewise = {}
        for key in extracted_data.keys():
            chunked_slidewise[key] = {}
            for page in extracted_data[key]['contents'].keys():
                chunk_text = f'<<beginning of OCR output>>\n\nSlide #{page} text:\n' + '\n'.join([para for para in extracted_data[key]['contents'][page]])
                
                if extracted_data[key].get('tables') is not None:
                    if extracted_data[key]['tables'].get(page) is not None:
                        chunk_text += f'\n\nSlide #{page} tables:\n---\n' + '\n---\n'.join(
                            ['\t|\n'.join(
                                ['|' + '\t|'.join(
                                    [cell for cell in row]
                                ) for row in table]
                            ) + '\t|\n' for table in extracted_data[key]['tables'][page]]
                        ) + '\n---\n'
                
                # Add footer to the chunk if available
                if extracted_data[key]['footers'].get(page):
                    chunk_text += f'\n\nSlide #{page} footer:\n' + extracted_data[key]['footers'][page]
                
                chunk_text += '\n\n<<end of OCR output>>'
                chunked_slidewise[key][page] = chunk_text
        return chunked_slidewise
    
def save_json(data, filename):
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def process_document(doc_path, extractor):
    try:
        print(f"PDF at {doc_path} processing...\n", end=' ')
        extracted_data = extractor.extract_text_n_tables(doc_path)
        doc_processed = extractor.prepare_to_chunks(extracted_data)
        print(f"Done! #Slides = {len(list(doc_processed[doc_path].keys()))}")
        return doc_path, doc_processed
    except Exception as e:
        print(f"Error processing {doc_path}: {e}")
        return doc_path, None

def doc_ingestor(doc_paths, chunked_slidewise={}):
    extractor = AzureDocProcessor()
    lock = threading.Lock()
    
    def thread_target(sub_batch):
        for doc_path in sub_batch:
            if doc_path in chunked_slidewise.keys():
                print(f"Skipping {doc_path} as it already exists in chunked_slidewise.")
                continue
            doc_path, doc_processed = process_document(doc_path, extractor)
            if doc_processed:
                with lock:
                    chunked_slidewise.update(doc_processed)
    
    sub_batch_size = 5
    sub_batches = list(batch_list(doc_paths, sub_batch_size))
    threads = []
    
    for sub_batch in sub_batches:
        thread = threading.Thread(target=thread_target, args=(sub_batch,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    return chunked_slidewise


chunked_slidewise = {}
import os
pdfs = ['TEST.pdf']
print(os.path.exists('TEST.pdf'))
chunked_slidewise = doc_ingestor(pdfs, chunked_slidewise)