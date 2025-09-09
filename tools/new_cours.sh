#!/usr/bin/env bash
set -euo pipefail
COURSE="${1:-}"
if [ -z "$COURSE" ]; then
  echo "Usage: ./tools/new_cours.sh CoursX"
  exit 1
fi

if [ -d "$COURSE" ]; then
  echo "Le dossier $COURSE existe déjà."
  exit 1
fi

mkdir -p "$COURSE"/{notebooks,src,data,tests}
touch "$COURSE/README.md" "$COURSE/.env"
echo "# $COURSE" > "$COURSE/README.md"

echo "✅ Structure créée pour $COURSE/"
echo ""
echo "Ajoute ce bloc dans EIAP.code-workspace (dans la clé 'folders') :"
echo '{ "name": "'$COURSE'", "path": "'$COURSE'" },'
