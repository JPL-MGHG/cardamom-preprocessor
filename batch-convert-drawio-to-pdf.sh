# Navigate to the project's root directory
cd /Users/shah/Desktop/Development/ghg/cardamom-preprocessor/

# Create a subdirectory for the PDFs inside the 'plans' directory
mkdir -p plans/pdfs

# Loop through all the lifecycle diagrams and convert them to PDF
for file in plans/*-Lifecycle.drawio.xml plans/*.drawio.xml plans/*.drawio; do
  # Check if the file exists to avoid errors with glob patterns
  if [ -f "$file" ]; then
    # Create a clean output filename without the multiple extensions
    output_name=$(basename "$file" .drawio.xml | sed 's/\.xml$//' | sed 's/\.drawio$//')
    echo "Converting $file to plans/pdfs/${output_name}.pdf"

    # Run the conversion command
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf -o "plans/pdfs/${output_name}.pdf" -- "$file"
  fi
done

echo "All diagrams have been converted to PDFs in the 'plans/pdfs' directory."

