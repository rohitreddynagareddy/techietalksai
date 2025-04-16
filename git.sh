
(base) Sreeprakashs-MacBook-Pro:techietalksai sree$ git branch --set-upstream-to=origin/main main
branch 'main' set up to track 'origin/main'.
(base) Sreeprakashs-MacBook-Pro:techietalksai sree$ 
(base) Sreeprakashs-MacBook-Pro:techietalksai sree$ git pull --no-rebase main
fatal: 'main' does not appear to be a git repository
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
(base) Sreeprakashs-MacBook-Pro:techietalksai sree$ git pull --no-rebase 
Auto-merging .gitignore
CONFLICT (content): Merge conflict in .gitignore
Auto-merging llm/pipeline/llm-pipeline/step1_train_custom/Dockerfile
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step1_train_custom/Dockerfile
Auto-merging llm/pipeline/llm-pipeline/step1_train_custom/docker-compose.yml
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step1_train_custom/docker-compose.yml
Auto-merging llm/pipeline/llm-pipeline/step1_train_custom/output_model/config.json
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step1_train_custom/output_model/config.json
Auto-merging llm/pipeline/llm-pipeline/step1_train_custom/output_model/generation_config.json
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step1_train_custom/output_model/generation_config.json
Auto-merging llm/pipeline/llm-pipeline/step2_inference_custom/Dockerfile
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step2_inference_custom/Dockerfile
Auto-merging llm/pipeline/llm-pipeline/step2_inference_custom/docker-compose.yml
CONFLICT (add/add): Merge conflict in llm/pipeline/llm-pipeline/step2_inference_custom/docker-compose.yml
Automatic merge failed; fix conflicts and then commit the result.
