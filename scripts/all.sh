cd gold_standard
./sft.sh

cd ../cdpo
python generate.py

cd ../introspection
python interaction.py
python reflection.py