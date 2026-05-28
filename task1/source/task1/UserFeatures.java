package task1;

import java.io.Serializable;

public class UserFeatures implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public long userId;
    public int totalBehaviors;
    public int buyCount;
    public double conversionRate;
    
    public UserFeatures() {}
    
    public UserFeatures(long userId, int totalBehaviors, int buyCount, double conversionRate) {
        this.userId = userId;
        this.totalBehaviors = totalBehaviors;
        this.buyCount = buyCount;
        this.conversionRate = conversionRate;
    }
}
