# Navigate to the directory
cd /Users/shah/Desktop/Development/ghg/cardamom-preprocessor/plans/

# Create a subdirectory for the images
mkdir -p images

# Loop through all the lifecycle diagrams and convert them
for file in *-Lifecycle.drawio.xml *.drawio; do
  # Create a clean output filename
  output_name=$(basename "$file" .drawio.xml | sed 's/\.xml$//')
  echo "Converting $file to images/${output_name}.png"

  # Run the conversion command
  /Applications/draw.io.app/Contents/MacOS/draw.io -x -f png -o "images/${output_name}.png" -- "$file"
done

echo "All diagrams have been converted to PNGs in the 'images' directory."

