namespace TravelAgents.Models;

public class City
{
    public string Name { get; set; } = string.Empty;
    public string Country { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public List<string> PopularAttractions { get; set; } = new();
    public string BestTimeToVisit { get; set; } = string.Empty;
}

public class Hotel
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public decimal PricePerNight { get; set; }
    public string Address { get; set; } = string.Empty;
    public int StarRating { get; set; }
    public List<string> Amenities { get; set; } = new();
    public string WebsiteUrl { get; set; } = string.Empty;
}

public class Activity
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public string Duration { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
}

public class ExtractedPreferences
{
    public string Budget { get; set; } = string.Empty;
    public string Dates { get; set; } = string.Empty;
    public List<string> Interests { get; set; } = new();
    public string TravelStyle { get; set; } = string.Empty;
}

public class TravelRecommendation
{
    public City? RecommendedCity { get; set; }
    public List<Hotel> RecommendedHotels { get; set; } = new();
    public List<Activity> RecommendedActivities { get; set; } = new();
    public string Summary { get; set; } = string.Empty;
    public ExtractedPreferences? ExtractedPreferences { get; set; }
} 