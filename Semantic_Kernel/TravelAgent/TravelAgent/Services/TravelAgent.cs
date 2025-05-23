using Microsoft.SemanticKernel;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TravelAgents.Models;

namespace TravelAgents.Services;

public class TravelAgent
{
    private readonly Kernel _kernel;
    private readonly ConversationMemory _memory;
    private readonly string _logFile = "travel_agent.log";
    
    private void LogToFile(string message)
    {
        try
        {
            File.AppendAllText(_logFile, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}{Environment.NewLine}");
        }
        catch
        {
            // Silently fail if we can't write to the log file
        }
    }
    
    private const string SystemPrompt = @"You are a travel agent AI assistant. When a user asks about travel recommendations, respond with a JSON object containing the following information:

{
    ""recommendedCity"": {
        ""name"": ""city name"",
        ""country"": ""country name"",
        ""description"": ""brief description"",
        ""popularAttractions"": [""attraction 1"", ""attraction 2""],
        ""bestTimeToVisit"": ""best time to visit""
    },
    ""recommendedHotels"": [
        {
            ""name"": ""accommodation name"",
            ""description"": ""accommodation description"",
            ""pricePerNight"": 100.00,
            ""starRating"": 3,
            ""amenities"": [""amenity 1"", ""amenity 2""],
            ""type"": ""accommodation type (hotel, motel, hostel, etc.)"",
            ""websiteUrl"": ""https://real-hotel-website.com"",
            ""address"": ""hotel address""
        }
    ],
    ""recommendedActivities"": [
        {
            ""name"": ""activity name"",
            ""description"": ""activity description"",
            ""price"": 50.00,
            ""duration"": ""duration"",
            ""category"": ""category""
        }
    ],
    ""summary"": ""brief summary of recommendations"",
    ""extractedPreferences"": {
        ""budget"": ""budget info"",
        ""dates"": ""travel dates"",
        ""interests"": [""interest 1"", ""interest 2""],
        ""travelStyle"": ""travel style""
    }
}

CRITICAL RULES FOR CONTEXT MAINTENANCE:
1. Respond ONLY with the JSON object - no other text before or after.

2. STRICT CITY CONTEXT RULE:
   - The most recently mentioned city MUST be maintained in all follow-up queries
   - NEVER change the city unless the user explicitly mentions a new city name
   - ALWAYS include the complete city information (name, country, description, attractions, best time)
   - For follow-up queries about specific aspects (hotels, activities), keep ALL previous city context

3. ACCOMMODATION RULES:
   - When user asks for specific types (hotels, motels, hostels, etc.), ONLY show that type
   - Provide REAL accommodation names and accurate descriptions for the CURRENT city
   - Include actual price ranges and star ratings
   - List genuine amenities (at least 3 per accommodation)
   - Focus on the specified area or preferences (e.g., city center, airport area)
   - When showing accommodations, always specify the type in the 'type' field
   - For motels, focus on roadside/highway locations and basic amenities
   - For hostels, focus on social areas and shared facilities
   - For hotels, include star ratings and full amenities
   - ALWAYS provide REAL, FUNCTIONAL website URLs for each accommodation (e.g., https://www.marriott.com/hotels/travel/nycwh-w-new-york-times-square/)
   - If you don't know the exact URL, use a search URL like https://www.google.com/search?q=HotelName+CityName+booking

4. PREFERENCE MAINTENANCE:
   - Consider the COMPLETE conversation history
   - Maintain budget preferences across all recommendations
   - Maintain location preferences (e.g., city center, airport)
   - Maintain dates and timing information
   - Maintain any mentioned interests or travel style
   - ALWAYS include previous preferences in extractedPreferences

5. QUALITY RULES:
   - Never use placeholder values
   - Provide specific, contextual information
   - Keep all previously provided correct information
   - Only update the specifically queried aspects
   - ALL prices must be numbers (not strings) with 2 decimal places
   - ALL star ratings must be whole numbers
   - Descriptions must be detailed and specific to the location
   - Include location-specific details in accommodation descriptions";

    private const string PreferenceExtractionPrompt = @"Analyze the following conversation and extract key travel preferences.

IMPORTANT: You must respond ONLY with a valid JSON object that follows this exact schema - no other text before or after:

{
    ""budget"": ""string"",
    ""dates"": ""string"",
    ""interests"": [""string""],
    ""travelStyle"": ""string""
}

Previous conversation:
{0}

Remember: Respond ONLY with the JSON object - no other text.";

    public TravelAgent(string apiKey, string endpoint, string deploymentName, string modelId)
    {
        try
        {
            Console.WriteLine("Creating kernel builder...");
            var builder = Kernel.CreateBuilder();
            
            Console.WriteLine("Adding Azure OpenAI chat completion...");
            builder.AddAzureOpenAIChatCompletion(
                deploymentName: deploymentName,
                endpoint: endpoint,
                modelId: modelId,
                apiKey: apiKey);
            
            Console.WriteLine("Building kernel...");
            _kernel = builder.Build();
            
            Console.WriteLine("Initializing conversation memory...");
            _memory = new ConversationMemory();
            
            Console.WriteLine("Travel Agent constructor completed successfully");
            
            // Clear the log file on startup
            try
            {
                File.WriteAllText(_logFile, $"Travel Agent started at {DateTime.Now}{Environment.NewLine}");
                Console.WriteLine($"Log file created at: {Path.GetFullPath(_logFile)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not create log file: {ex.Message}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in TravelAgent constructor: {ex.GetType().Name}");
            Console.WriteLine($"Error message: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner error: {ex.InnerException.Message}");
            }
            throw;
        }
    }

    private string ExtractJsonFromText(string text)
    {
        LogToFile($"Extracting JSON from text (length: {text?.Length ?? 0})");
        
        if (string.IsNullOrEmpty(text))
        {
            LogToFile("Input text is null or empty");
            return "{}";
        }

        int startIndex = text.IndexOf('{');
        int endIndex = text.LastIndexOf('}');
        
        LogToFile($"JSON indices: start={startIndex}, end={endIndex}");
        
        if (startIndex == -1 || endIndex == -1 || endIndex <= startIndex)
        {
            LogToFile("Invalid JSON structure detected");
            return "{}";
        }
        
        var extractedJson = text.Substring(startIndex, endIndex - startIndex + 1);
        LogToFile($"Extracted JSON: {extractedJson}");
        
        return extractedJson;
    }

    public async Task<TravelRecommendation> GetStructuredRecommendationAsync(string userQuery)
    {
        try
        {
            LogToFile($"Processing user query: {userQuery}");
            _memory.AddMessage("user", userQuery);

            // First, extract preferences from the conversation
            var conversationHistory = _memory.GetConversationHistory();
            LogToFile($"Conversation history: {conversationHistory}");
            
            var preferencesPrompt = PreferenceExtractionPrompt.Replace("{0}", conversationHistory);
            var preferencesResult = await _kernel.InvokePromptAsync(preferencesPrompt);
            var preferencesResponse = preferencesResult.GetValue<string>() ?? "{}";
            LogToFile($"Raw preferences response: {preferencesResponse}");
            
            preferencesResponse = ExtractJsonFromText(preferencesResponse);
            LogToFile($"Extracted preferences: {preferencesResponse}");
            
            // Then, get the travel recommendation with enhanced context
            var contextPrompt = $"{SystemPrompt}\n\nCurrent conversation history:\n{conversationHistory}\n\nExtracted preferences: {preferencesResponse}\n\nCurrent user query: {userQuery}\n\nRemember to maintain the city context and all preferences from the conversation history.";
            
            var result = await _kernel.InvokePromptAsync(contextPrompt);
            var response = result.GetValue<string>() ?? "{}";
            LogToFile($"Raw AI response: {response}");
            
            response = ExtractJsonFromText(response);
            LogToFile($"Extracted JSON: {response}");
            
            _memory.AddMessage("assistant", response);

            try
            {
                // Parse the valid JSON with number handling
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    AllowTrailingCommas = true,
                    NumberHandling = JsonNumberHandling.AllowReadingFromString
                };

                var parsedRecommendation = JsonSerializer.Deserialize<TravelRecommendation>(response, options);
                if (parsedRecommendation != null)
                {
                    LogToFile("Successfully parsed recommendation");
                    
                    // Fill in any null collections with empty lists
                    parsedRecommendation.RecommendedHotels ??= new List<Hotel>();
                    parsedRecommendation.RecommendedActivities ??= new List<Activity>();
                    parsedRecommendation.ExtractedPreferences ??= new ExtractedPreferences
                    {
                        Interests = new List<string>()
                    };
                    
                    if (parsedRecommendation.RecommendedCity != null)
                    {
                        parsedRecommendation.RecommendedCity.PopularAttractions ??= new List<string>();
                    }
                    
                    return parsedRecommendation;
                }
            }
            catch (JsonException ex)
            {
                LogToFile($"JSON parsing error: {ex.Message}");
                LogToFile($"Problematic JSON: {response}");
                return new TravelRecommendation
                {
                    RecommendedCity = new City
                    {
                        Name = "Error Processing Response",
                        Country = "Unknown",
                        Description = $"We encountered an error processing your request: {ex.Message}",
                        PopularAttractions = new List<string>(),
                        BestTimeToVisit = "Unknown"
                    },
                    RecommendedHotels = new List<Hotel>(),
                    RecommendedActivities = new List<Activity>(),
                    Summary = "I apologize, but I couldn't process your request properly. Please try rephrasing your question.",
                    ExtractedPreferences = new ExtractedPreferences { Interests = new List<string>() }
                };
            }

            return new TravelRecommendation
            {
                RecommendedCity = new City
                {
                    Name = "Error Processing City",
                    Country = "Unknown",
                    Description = "We encountered an error processing your request.",
                    PopularAttractions = new List<string>(),
                    BestTimeToVisit = "Unknown"
                },
                RecommendedHotels = new List<Hotel>(),
                RecommendedActivities = new List<Activity>(),
                Summary = "An error occurred while processing your request. Please try again.",
                ExtractedPreferences = new ExtractedPreferences { Interests = new List<string>() }
            };
        }
        catch (Exception ex)
        {
            LogToFile($"Unexpected error: {ex.Message}");
            LogToFile($"Stack trace: {ex.StackTrace}");
            return new TravelRecommendation
            {
                RecommendedCity = new City
                {
                    Name = "Error",
                    Country = "Unknown",
                    Description = "An unexpected error occurred.",
                    PopularAttractions = new List<string>(),
                    BestTimeToVisit = "Unknown"
                },
                RecommendedHotels = new List<Hotel>(),
                RecommendedActivities = new List<Activity>(),
                Summary = $"I apologize, but an unexpected error occurred: {ex.Message}",
                ExtractedPreferences = new ExtractedPreferences { Interests = new List<string>() }
            };
        }
    }
} 