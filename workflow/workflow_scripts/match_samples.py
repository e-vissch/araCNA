import requests

fields = [
    "md5sum",
    "file_name",
    "cases.samples.sample_id",
    "cases.samples.sample_type",
    "cases.case_id",
]

fields = ",".join(fields)

cases_endpt = "https://api.gdc.cancer.gov/cases"


filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "md5sum", "value": ["put md codes here"]}},
        {"op": "in", "content": {"field": "files.data_format", "value": ["BAM"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["WGS"]}},
    ],
}


files_endpt = "https://api.gdc.cancer.gov/files"

# With a GET request, the filters parameter needs to be converted
# from a dictionary to JSON-formatted string
# A POST is used, so the filter parameters can be passed directly as a Dict object.
params = {"filters": filters, "fields": fields, "format": "TSV", "size": "2000"}

# The parameters are passed to 'json' rather than 'params' in this case
response = requests.post(
    files_endpt, headers={"Content-Type": "application/json"}, json=params
)

decoded_content = response.content.decode("utf-8")
normalized_content = decoded_content.replace("\r\n", "\n")

print(normalized_content)
