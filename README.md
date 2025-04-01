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
    ) \(
      (.results // "" | strip)
    ) \(
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
' pubmed_Data/data.json > pubmed_data/data.txt
```

load env variables
```
source /Users/jason/Desktop/projects/streamlit-graphrag/env.sh
```

Run locally

```sh
poetry run python app.py --question="What are the some examples of the risk factors for atherosclerosis?"
```

