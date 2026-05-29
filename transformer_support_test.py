from fastembed import TextEmbedding
import pandas as pd

supported = pd.DataFrame(TextEmbedding.list_supported_models())
print(supported[['model', 'dim', 'description', 'size_in_GB']])