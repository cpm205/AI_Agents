using Microsoft.SemanticKernel;

namespace TravekAgent.Models;

public class ConversationMemory
{
    private readonly int _maxMessages = 10;
    public List<(string Role, string Content)> Messages { get; } = new();
    public Dictionary<string, string> UserPreferences { get; } = new();

    public void AddMessage(string role, string content)
    {
        Messages.Add((role, content));
        
        // Keep only the most recent messages to avoid context overflow
        if (Messages.Count > _maxMessages)
        {
            Messages.RemoveAt(0);
        }
    }

    public void UpdatePreferences(string key, string value)
    {
        if (string.IsNullOrEmpty(value))
            return;
            
        UserPreferences[key] = value;
    }

    public string GetConversationHistory()
    {
        if (Messages.Count == 0)
            return "No conversation history yet.";
            
        return string.Join("\n", Messages.Select(m => $"{m.Role}: {m.Content}"));
    }

    public string GetPreferencesSummary()
    {
        if (UserPreferences.Count == 0)
            return "No preferences extracted yet.";
            
        return string.Join("\n", UserPreferences.Select(p => $"{p.Key}: {p.Value}"));
    }
    
    public void Clear()
    {
        Messages.Clear();
        UserPreferences.Clear();
    }
    
    public bool HasPreference(string key)
    {
        return UserPreferences.ContainsKey(key) && !string.IsNullOrEmpty(UserPreferences[key]);
    }
    
    public string GetPreference(string key, string defaultValue = "")
    {
        return UserPreferences.TryGetValue(key, out var value) ? value : defaultValue;
    }
} 