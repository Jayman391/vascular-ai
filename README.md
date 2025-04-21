preprocess pubmed json data into strings for vector store 

```sh
jq -r '
  def re: "[\n\r]+";             # Define the regex first
  def strip: gsub(re; " ");      # Then use it in the strip function

  to_entries[]
  | .value
  | select((.abstract // "" | length > 0) or (.results // "" | length > 0) or (.full_text // "" | length > 0))
  | "\(
      (.title // "" | strip)
    ) \(
      (.abstract // "" | strip)
    )  \(
    )  \(
      (.authors // []
        | map(
            (.affiliation // "" | strip) + " " +
            (.firstname // "" | strip) + " " +
            (.initials // "" | strip) + " " +
            (.lastname // "" | strip)
          )
        | join(", ")
      )
    ) \(
      (.keywords // "" | strip)
    )"
' pubmed_Data/vascular_data_pubmed.json > pubmed_data/vascular_data_pubmed.txt
```

now we must do the same for our jvs data, which is a bunch of pdfs in compressed format

```sh

# append string to data/vascular_data_jvs.txt
# Iterate through all compressed files in data/vascular_data_jvs
for compressed_file in data/vascular_data_jvs/*.zip; do
  # Unzip the compressed file into a temporary folder
  unzip "$compressed_file" -d temp_unzipped_folder
  # Iterate through all PDF files in the unzipped folder
  for pdf_file in temp_unzipped_folder/*.pdf; do
    # Extract text from the PDF file and store it in a variable
    extracted_text=$(pdftotext "$pdf_file" -)
    # Make the whole article be one line by replacing newlines with spaces
    processed_text=$(echo "$extracted_text" | tr '\n' ' ')
    # Append the processed text to the output file
    echo "$processed_text" >> data/vascular_data_jvs.txt
  done
  # Clean up the temporary folder
  rm -rf temp_unzipped_folder
done
```

load env variables
```
export $(grep -v '^#' .env | xargs)
```

Run locally

```sh
poetry run python app.py
```

In a separate terminal, test functionality via cURL request

```sh
curl -X POST http://127.0.0.1:5000/query \
-H "Content-Type: application/json" \
-d '{"query": "How would one treat vascular ehlers danlos syndrome?"}'
```