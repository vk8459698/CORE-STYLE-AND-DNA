import json
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass
import os
from collections import defaultdict
import re

@dataclass
class Song:
    """Individual song data structure - simplified"""
    tags: List[str]  # Genre/style tags
    classification: str  # Song classification/type
    
@dataclass
class CoreStyle:
    """Core style data structure"""
    name: str
    description: str
    tags: List[str]
    signature_sound_tags: List[str]
    songs: List[int]  # indices of songs in this style

@dataclass
class ArtistDNA:
    """Artist DNA level summary"""
    dna_tags: List[str]

class OpenAIMusicCategorizer:
    """
    Music style categorization system using OpenAI API
    Creates core styles, signature sounds, and DNA-level tags
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        # Set up OpenAI client with new API format
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key must be provided either as parameter or environment variable OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            
        # Use a model that supports JSON mode
        self.model = model
        if model == "gpt-4":
            print("  Note: Switching to gpt-4o for JSON response format support")
            self.model = "gpt-4o"
            
        self.songs = []
        self.core_styles = []
        self.artist_dna = None
        
    def add_song(self, tags: List[str], classification: str):
        """Add a song to the collection with simplified input"""
        song = Song(tags, classification)
        self.songs.append(song)
        
    def extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from response text, handling cases where it might not be pure JSON"""
        try:
            # First try to parse as pure JSON
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the response
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return None
            print(f"Could not extract JSON from response: {response_text[:200]}...")
            return None
        
    def call_openai_api(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.3):
        """Make API call to OpenAI using new client format"""
        try:
            # Try with JSON mode first
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as json_error:
                print(f"JSON mode failed, falling back to regular mode: {json_error}")
                # Fallback to regular mode without JSON format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return None
            
    def classify_songs_into_core_styles(self, artist_name: str = "Artist") -> Dict[str, Any]:
        """Use OpenAI to classify songs into core styles (1-10 max)"""
        
        # Prepare song data for OpenAI
        songs_data = []
        for i, song in enumerate(self.songs):
            song_info = {
                "song_id": i,
                "tags": song.tags,
                "classification": song.classification
            }
            songs_data.append(song_info)
            
        system_prompt = """You are a music industry expert specializing in artist style analysis. Your task is to analyze songs and group them into core musical styles (minimum 1, maximum 10 core styles).

IMPORTANT GUIDELINES:
- DO NOT be overly sensitive with categorization
- TRY TO COMBINE similar styles of music rather than creating too many separate categories
- Focus on major stylistic differences, not minor variations
- Group songs that share similar musical DNA, production style, or thematic elements
- Each core style should have at least 1-2 songs minimum

You must return a JSON response with this exact structure:
{
  "core_styles": [
    {
      "style_name": "Style Name",
      "song_ids": [0, 1, 2],
      "primary_characteristics": ["char1", "char2", "char3"]
    }
  ]
}

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        user_prompt = f"""Analyze these {len(songs_data)} songs for artist "{artist_name}" and group them into core musical styles.

Songs to analyze:
{json.dumps(songs_data, indent=2)}

Remember:
- Combine similar styles, don't be overly sensitive
- Maximum 10 core styles, minimum 1
- Focus on major musical differences
- Each style needs clear musical identity
- Return only valid JSON, no additional text"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=1500)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result
            else:
                print("Error parsing OpenAI response")
                return None
        return None
        
    def generate_core_style_details(self, style_name: str, song_ids: List[int], 
                                  primary_characteristics: List[str], artist_name: str) -> Dict[str, Any]:
        """Use OpenAI to generate detailed core style information"""
        
        # Get songs for this style
        style_songs = [self.songs[i] for i in song_ids]
        songs_info = []
        
        for i, song in enumerate(style_songs):
            songs_info.append({
                "tags": song.tags,
                "classification": song.classification
            })
            
        system_prompt = f"""You are a music industry expert writing detailed style analysis. Create comprehensive details for a core musical style.

You must return a JSON response with this exact structure:
{{
  "style_name": "Final Style Name",
  "description": "Detailed 200-400 character description in the style: '{artist_name}'s [STYLE] vision merges [key elements]. The productions focus on [characteristics]. By [approach], {artist_name} [impact/contribution].'",
  "core_style_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "signature_sound_tags": ["sound1", "sound2", "sound3", "sound4", "sound5"]
}}

The description should sound professional and music-industry focused, similar to how you'd describe an artist's style in a music magazine or streaming platform.

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        user_prompt = f"""Create detailed information for this core style:

Style Name: {style_name}
Artist: {artist_name}
Primary Characteristics: {primary_characteristics}
Number of Songs: {len(song_ids)}

Songs in this style:
{json.dumps(songs_info, indent=2)}

Generate:
1. A refined style name (if needed)
2. A compelling 200-400 character description following the format shown
3. 5 core style tags (genre/feel descriptors)
4. 5 signature sound tags (sonic characteristics)

Return only valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=1200)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result
            else:
                print(f"Error parsing style details for {style_name}")
                return None
        return None
        
    def generate_artist_dna_tags(self, core_styles_data: List[Dict], artist_name: str) -> List[str]:
        """Use OpenAI to generate 5 DNA-level tags summarizing all core styles"""
        
        system_prompt = """You are a music industry expert. Generate 5 super-tags that summarize an artist's overall musical DNA based on all their core styles.

These DNA tags should capture the artist's overarching musical identity across all styles.

You must return a JSON response:
{
  "dna_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}

IMPORTANT: Your response must be valid JSON only, no additional text or explanations."""

        # Prepare core styles data
        styles_summary = []
        for style in core_styles_data:
            style_info = {
                "name": style.get('style_name', ''),
                "core_tags": style.get('core_style_tags', []),
                "signature_tags": style.get('signature_sound_tags', [])
            }
            styles_summary.append(style_info)
            
        user_prompt = f"""Generate 5 DNA-level tags for artist "{artist_name}" based on these core styles:

{json.dumps(styles_summary, indent=2)}

The DNA tags should represent the artist's overall musical identity that spans across all their core styles.

Return only valid JSON, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.call_openai_api(messages, max_tokens=300)
        
        if response:
            result = self.extract_json_from_response(response)
            if result:
                return result.get('dna_tags', [])
            else:
                print("Error parsing DNA tags response")
                return []
        return []
        
    def analyze_artist(self, artist_name: str = "Artist") -> Dict[str, Any]:
        """Complete analysis pipeline using OpenAI API"""
        
        if not self.songs:
            return {"error": "No songs provided for analysis"}
            
        print(f" Analyzing {len(self.songs)} songs for {artist_name}...")
        print(f" Using model: {self.model}")
        
        # Step 1: Classify songs into core styles
        print(" Classifying songs into core styles...")
        classification_result = self.classify_songs_into_core_styles(artist_name)
        
        if not classification_result:
            return {"error": "Failed to classify songs into core styles"}
            
        core_styles_data = []
        
        # Step 2: Generate detailed information for each core style
        print(" Generating detailed core style information...")
        for style_info in classification_result.get('core_styles', []):
            style_name = style_info.get('style_name', '')
            song_ids = style_info.get('song_ids', [])
            characteristics = style_info.get('primary_characteristics', [])
            
            print(f"   Processing: {style_name}")
            
            style_details = self.generate_core_style_details(
                style_name, song_ids, characteristics, artist_name
            )
            
            if style_details:
                style_details['song_ids'] = song_ids
                style_details['song_count'] = len(song_ids)
                core_styles_data.append(style_details)
                
        # Step 3: Generate Artist DNA tags
        print(" Generating Artist DNA tags...")
        dna_tags = self.generate_artist_dna_tags(core_styles_data, artist_name)
        
        # Format final result
        result = {
            "artist_name": artist_name,
            "total_songs_analyzed": len(self.songs),
            "total_core_styles": len(core_styles_data),
            "core_styles": core_styles_data,
            "artist_dna": {
                "dna_tags": dna_tags
            }
        }
        
        print(" Analysis complete!")
        return result
        
    def print_analysis_results(self, results: Dict[str, Any]):
        """Print results in a formatted way"""
        
        if "error" in results:
            print(f" Error: {results['error']}")
            return
            
        print("=" * 80)
        print(f" {results['artist_name']} - AI-POWERED MUSIC DNA ANALYSIS")
        print("=" * 80)
        print(f" Songs Analyzed: {results['total_songs_analyzed']}")
        print(f" Core Styles Found: {results['total_core_styles']}")
        print("=" * 80)
        
        print("\n CORE STYLES:")
        print("-" * 60)
        
        for i, style in enumerate(results['core_styles'], 1):
            print(f"\n[{i}] {style['style_name'].upper()}")
            print(f"     {results['artist_name']}")
            print(f"     {style['description']}")
            print(f"      Core Tags: {', '.join(style['core_style_tags'])}")
            print(f"     Signature Sound: {', '.join(style['signature_sound_tags'])}")
            print(f"     Songs: {style['song_count']} tracks")
            print("-" * 60)
        
        print(f"\n ARTIST DNA TAGS:")
        print("-" * 60)
        print(f"    {', '.join(results['artist_dna']['dna_tags'])}")
        
        print("\n" + "=" * 80)

    def save_results_to_json(self, results: Dict[str, Any], filename: str = "music_analysis_results.json"):
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f" Results saved to {filename}")
        except Exception as e:
            print(f" Error saving results: {e}")

# Example usage and main execution
def example_usage():
    """Example of how to use the OpenAI Music Categorizer with simplified input"""
    
    # Initialize with your OpenAI API key
    try:
        categorizer = OpenAIMusicCategorizer(model="gpt-4o")
    except ValueError as e:
        print(f" {e}")
        print(" Set your OpenAI API key as environment variable OPENAI_API_KEY")
        print(" Or pass it directly: OpenAIMusicCategorizer(api_key='your-key')")
        return None, None
    
    # Add sample songs with simplified input (just tags and classification)
    sample_songs = [
        {
            "tags": ["EDM", "electronic", "dance", "festival", "high-energy", "synth"],
            "classification": "Electronic Dance Music"
        },
        {
            "tags": ["progressive house", "electronic", "melodic", "atmospheric", "emotional"],
            "classification": "Progressive House"
        },
        {
            "tags": ["folktronica", "acoustic", "electronic", "chill", "organic", "indie"],
            "classification": "Electronic Folk"
        },
        {
            "tags": ["electro-pop", "synth-pop", "upbeat", "bright", "catchy", "commercial"],
            "classification": "Pop Electronic"
        },
        {
            "tags": ["ambient", "electronic", "instrumental", "textured", "atmospheric"],
            "classification": "Ambient Electronic"
        },
        {
            "tags": ["big room", "house", "festival", "energetic", "anthem", "mainstream"],
            "classification": "Big Room House"
        },
        {
            "tags": ["techno", "minimal", "driving", "industrial", "underground"],
            "classification": "Minimal Techno"
        },
        {
            "tags": ["trap", "hip-hop", "bass", "electronic", "urban", "beats"],
            "classification": "Electronic Trap"
        }
    ]
    
    # Add songs to categorizer
    for song_data in sample_songs:
        categorizer.add_song(
            tags=song_data["tags"],
            classification=song_data["classification"]
        )
    
    # Analyze with OpenAI
    results = categorizer.analyze_artist(artist_name="ASH POURNOURI")
    
    # Print formatted results
    categorizer.print_analysis_results(results)
    
    # Save results to JSON file
    if results and "error" not in results:
        categorizer.save_results_to_json(results)
    
    return categorizer, results

def main():
    """Main function to run the music analysis"""
    print(" Music Style Classifier with OpenAI (Simplified Input)")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print(" OpenAI API key not found!")
        print(" Please set your API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   OR modify the code to pass it directly")
        return
    
    # Run example analysis
    categorizer, results = example_usage()
    
    if categorizer and results and "error" not in results:
        print(f"\n Analysis completed successfully!")
        print(f" Check 'music_analysis_results.json' for detailed results")
    else:
        print("❌ Analysis failed. Please check your API key and try again.")

if __name__ == "__main__":
    main()