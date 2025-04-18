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
' pubmed_Data/data.json > pubmed_data/vascular_data.txt
```

load env variables
```
source env.sh
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