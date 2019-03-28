# PFI2019
Protein Function Inference via Artificial Intelligence
On a molecular scale, most diseases are caused by the malfunction of proteins. This makes understanding every proteins function crucial for understanding disease. Unfortunately, measureing the funciton of a protein is extremley difficult. To inferr function, protien-protein interactions (PPI) are used along with a few methods of machine learning.

<b>Setting up the env.</b><br>
Recommendations:<br>
  Linux<br>
  GPU<br>
Languages:<br>
  Python<br>
  Perl<br>
  YAML<br>
  Bash<br>
Dependencys:<br>
  Tensorflow<br>
  Numpy<br>
  SKLearn<br>
  Matplotlib<br>
<br>
<b>Obtaining The PPI</b>
All data file missing, because they are to large to store on github. Luckily they are easily obtainable. Follow steps below to create an up to data version of them.
  1. Got to https://string-db.org/cgi/download.pl?sessionId=40Cx3XVALsca&species_text=9606
  2. Download 9606.protein.links.full.v10.5.txt.gz and uncompress. (this is human ppi data)
  3. mkdir 'InteractionData', then move the ppi to this folder.
  4. cat InteractionData/9606.protein.links.full.v10.5.txt | perl -wan PPI_parse.pl > Protein-Protein_Combined-Interactions.txt
  5. mkdir data
  6. Using Jupyter notebook, run preprocess_data_v3.ipynb.
  7. The PPI will appear as a serialzed numpy binary array in the folder data.
  
Obtaining the GO:
  Coming Soon
