### How to
- Unzip `merged-cleansed.7z` in the root folder as `merged-cleansed.json`.
- Run the script `topic_analysis.py`.
- Once completed it will ouput LDA model and will open HTML view on `http://localhost:8888`.

### Files
* `merged-cleansed.7z` - Formatted largely cleansed twitter dataset. When unpacked it would be around ~400mb.
* `/analysis/` - Files relevant to analysis.
* `/analysis/models/` - Results from previous analysis.
    * `meta` - Containers metadata / params used to run the model.
* `/analysis/custom-stopwords.txt` - list of custom stop words.
