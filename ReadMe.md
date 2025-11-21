#create a new repository on the command line
echo "# soft-computing-lab" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/16ajeet/soft-computing-lab.git
git push -u origin main

push an existing repository from the command line
git remote add origin https://github.com/16ajeet/soft-computing-lab.git
git branch -M main
git push -u origin main