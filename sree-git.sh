  551  git filter-repo --path llm/pipeline/llm-pipeline/step0_building_base_model_v2/tiny_transformer.pth --invert-paths
  552  git commit -m "Remove large file from tracking"
  553  git rm --cached llm/pipeline/llm-pipeline/step0_building_base_model_v2/tiny_transformer.pth
  554  git filter-repo --path llm/pipeline/llm-pipeline/step0_building_base_model_v2/tiny_transformer.pth --invert-paths
  555  git push --force
  556  git remote add origin git@github.com:schogini/techietalksai.git
  557  git push --force
  558  sh push.sh llm
  559  history| tail
  560  history| tail > sree-git.sh
