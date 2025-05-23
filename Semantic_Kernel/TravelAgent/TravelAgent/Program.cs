using System.Text.Json;
using TravelAgents.Models;
using TravelAgents.Services;
using Microsoft.Extensions.Configuration;

// Create a cancellation token source that we can use to keep the program running
using var cts = new CancellationTokenSource();

try
{
    Console.WriteLine("Loading configuration...");
    // Load configuration from appsettings.json
    var configuration = new ConfigurationBuilder()
        .SetBasePath(Directory.GetCurrentDirectory())
        .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
        .Build();

    Console.WriteLine("Reading Azure OpenAI configuration...");
    var azureOpenAIConfig = configuration.GetSection("AzureOpenAI");
    var endpoint = azureOpenAIConfig["Endpoint"] ?? "https://devdemo.openai.azure.com/";
    var deploymentName = azureOpenAIConfig["DeploymentName"] ?? "gpt-35-turbo-16k";
    var modelId = azureOpenAIConfig["ModelId"] ?? "gpt-35-turbo-16k";
    var apiKey = azureOpenAIConfig["ApiKey"];

    Console.WriteLine($"Endpoint: {endpoint}");
    Console.WriteLine($"Deployment Name: {deploymentName}");
    Console.WriteLine($"Model ID: {modelId}");
    Console.WriteLine($"API Key length: {apiKey?.Length ?? 0}");

    if (string.IsNullOrEmpty(apiKey) || apiKey == "your-api-key-here")
    {
        Console.WriteLine("Error: Please update the ApiKey in appsettings.json with your Azure OpenAI API key.");
        WaitForExit();
        return;
    }

    TravelAgent? travelAgent = null;

    Console.WriteLine("Initializing Travel Agent...");
    travelAgent = new TravelAgent(apiKey, endpoint, deploymentName, modelId);
    Console.WriteLine("Travel Agent initialized successfully");

    Console.WriteLine("\nWelcome to the AI Travel Agent! I can help you plan your next adventure.");
    Console.WriteLine("Please describe your travel preferences (e.g., 'I want to visit a beach destination in Europe for 5 days in July, budget around $2000')");
    Console.WriteLine("Type 'exit' to quit, 'help' for assistance, or 'clear' to start a new conversation.");

    // Run the travel agent and wait for it to complete
    await RunTravelAgent(travelAgent);
}
catch (Exception ex)
{
    Console.WriteLine($"\nError during initialization: {ex.GetType().Name}");
    Console.WriteLine($"Error message: {ex.Message}");
    Console.WriteLine($"Stack trace: {ex.StackTrace}");
    if (ex.InnerException != null)
    {
        Console.WriteLine($"Inner error: {ex.InnerException.Message}");
    }
    WaitForExit();
    return;
}

// Helper method to wait for user input before exiting
void WaitForExit()
{
    Console.WriteLine("\nPress Enter to exit...");
    Console.ReadLine();
}

async Task RunTravelAgent(TravelAgent agent)
{
    try
    {
        while (!cts.Token.IsCancellationRequested)
        {
            Console.Write("\nYou: ");
            var userInput = await Task.Run(() => Console.ReadLine(), cts.Token);

            if (string.IsNullOrEmpty(userInput))
                continue;

            if (userInput.ToLower() == "exit")
            {
                cts.Cancel();
                break;
            }

            if (userInput.ToLower() == "help")
            {
                DisplayHelp();
                continue;
            }

            if (userInput.ToLower() == "clear")
            {
                Console.Clear();
                Console.WriteLine("Welcome to the AI Travel Agent! I can help you plan your next adventure.");
                Console.WriteLine("Please describe your travel preferences (e.g., 'I want to visit a beach destination in Europe for 5 days in July, budget around $2000')");
                Console.WriteLine("Type 'exit' to quit, 'help' for assistance, or 'clear' to start a new conversation.");
                continue;
            }

            try
            {
                Console.WriteLine("\nProcessing your request...");
                var recommendation = await agent.GetStructuredRecommendationAsync(userInput);
                DisplayRecommendation(recommendation);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine("Please try again or type 'help' for assistance.");
            }
        }
    }
    finally
    {
        Console.WriteLine("\nThank you for using the AI Travel Agent. Have a great trip!");
        WaitForExit();
    }
}

void DisplayHelp()
{
    Console.WriteLine("\n=== AI Travel Agent Help ===");
    Console.WriteLine("I can help you plan your next adventure by providing personalized travel recommendations.");
    Console.WriteLine("\nYou can ask me about:");
    Console.WriteLine("• Specific destinations (e.g., 'Tell me about Paris')");
    Console.WriteLine("• Travel preferences (e.g., 'I want a beach vacation in Europe')");
    Console.WriteLine("• Hotels (e.g., 'Recommend hotels in New York')");
    Console.WriteLine("• Activities (e.g., 'What can I do in Tokyo?')");
    Console.WriteLine("• Budget considerations (e.g., 'I have a budget of $2000 for a week')");
    Console.WriteLine("\nCommands:");
    Console.WriteLine("• 'help' - Display this help message");
    Console.WriteLine("• 'clear' - Start a new conversation");
    Console.WriteLine("• 'exit' - Exit the application");
    Console.WriteLine("\nFeel free to ask follow-up questions to refine your recommendations!");
}

void DisplayRecommendation(TravelRecommendation recommendation)
{
    Console.WriteLine("\nAI Travel Agent: Here's your personalized travel recommendation:");
    
    if (recommendation.RecommendedCity != null)
    {
        Console.WriteLine("\n📍 Recommended City:");
        Console.WriteLine($"   {recommendation.RecommendedCity.Name}, {recommendation.RecommendedCity.Country}");
        Console.WriteLine($"   {recommendation.RecommendedCity.Description}");
        
        if (recommendation.RecommendedCity.PopularAttractions?.Any() == true)
        {
            Console.WriteLine("\n   Popular Attractions:");
            foreach (var attraction in recommendation.RecommendedCity.PopularAttractions)
            {
                Console.WriteLine($"   • {attraction}");
            }
        }
        
        if (!string.IsNullOrEmpty(recommendation.RecommendedCity.BestTimeToVisit))
        {
            Console.WriteLine($"\n   Best Time to Visit: {recommendation.RecommendedCity.BestTimeToVisit}");
        }
    }

    if (recommendation.RecommendedHotels?.Any() == true)
    {
        Console.WriteLine("\n🏨 Recommended Hotels:");
        foreach (var hotel in recommendation.RecommendedHotels)
        {
            Console.WriteLine($"\n   {hotel.Name} ({hotel.StarRating}★)");
            Console.WriteLine($"   {hotel.Description}");
            Console.WriteLine($"   Address: ${hotel.Address}");
            if (!string.IsNullOrEmpty(hotel.WebsiteUrl))
            {
                Console.WriteLine($"   Website: {hotel.WebsiteUrl}");
            }
            
            if (hotel.Amenities?.Any() == true)
            {
                Console.WriteLine("   Amenities:");
                foreach (var amenity in hotel.Amenities)
                {
                    Console.WriteLine($"   • {amenity}");
                }
            }
        }
    }

    if (recommendation.RecommendedActivities?.Any() == true)
    {
        Console.WriteLine("\n🎯 Recommended Activities:");
        foreach (var activity in recommendation.RecommendedActivities)
        {
            Console.WriteLine($"\n   {activity.Name} ({activity.Category})");
            Console.WriteLine($"   {activity.Description}");
            Console.WriteLine($"   Duration: {activity.Duration}");
            Console.WriteLine($"   Price: ${activity.Price:N2}");
        }
    }

    if (!string.IsNullOrEmpty(recommendation.Summary))
    {
        Console.WriteLine("\n📝 Summary:");
        Console.WriteLine($"   {recommendation.Summary}");
    }

    if (recommendation.ExtractedPreferences != null)
    {
        Console.WriteLine("\n🎯 Extracted Preferences:");
        Console.WriteLine($"   Budget: {recommendation.ExtractedPreferences.Budget}");
        Console.WriteLine($"   Dates: {recommendation.ExtractedPreferences.Dates}");
        Console.WriteLine($"   Travel Style: {recommendation.ExtractedPreferences.TravelStyle}");
        
        if (recommendation.ExtractedPreferences.Interests?.Any() == true)
        {
            Console.WriteLine("   Interests:");
            foreach (var interest in recommendation.ExtractedPreferences.Interests)
            {
                Console.WriteLine($"   • {interest}");
            }
        }
    }
    
    Console.WriteLine("\nYou can ask follow-up questions to refine your recommendations or type 'help' for assistance.");
}
