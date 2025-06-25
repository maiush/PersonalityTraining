cd acr
./acr.sh glm-4-9b-it sarcasm 5
./acr.sh glm-4-9b-it nonchalance 5
./acr.sh glm-4-9b-it sycophancy 5

cd ../dpo
./glm-4-9b-it.sh glm-4-9b-it sarcasm 5
./glm-4-9b-it.sh glm-4-9b-it nonchalance 5
./glm-4-9b-it.sh glm-4-9b-it sycophancy 5

cd ../acr
./all.sh

cd ../dpo
./all.sh