% 数据准备
intersections = 1:5;
town01 = [78.26 88.64 83.48 18.65 68.23]; % 第一行数据
town10 = [61.8 31.68 48.63 1.71 5.56];    % 第二行数据

% 创建抗锯齿图形（关键设置）
figure('Position', [100 100 900 600], 'Color', 'w', 'Renderer', 'opengl')

% 绘制柱状图（解决锯齿关键设置）
bar_data = [town01; town10]';
h = bar(intersections, bar_data, 0.8, 'grouped', 'EdgeColor', 'none');

% 精确HEX颜色设置
hex_orange = '#EE822F'; % Town01
hex_blue = '#4874CB';   % Town10
rgb_orange = sscanf(hex_orange(2:end),'%2x%2x%2x',[1 3])/255;
rgb_blue = sscanf(hex_blue(2:end),'%2x%2x%2x',[1 3])/255;

% 柱状图样式
h(1).FaceColor = rgb_orange;
h(2).FaceColor = rgb_blue;

% 数据标签设置
label_props = {'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Arial',...
               'HorizontalAlignment', 'center'};
for i = 1:length(intersections)
    text(intersections(i)-0.2, town01(i)+2, sprintf('%.1f%%', town01(i)),...
         'Color', rgb_orange, label_props{:});
    text(intersections(i)+0.2, town10(i)+2, sprintf('%.1f%%', town10(i)),...
         'Color', rgb_blue, label_props{:});
end

% 坐标轴抗锯齿设置（解决锯齿朝外问题）
ax = gca;
set(ax, 'FontSize', 13, 'LineWidth', 1.5, 'XTick', intersections,...
        'TickDir', 'in', ...  % 关键修改：刻度朝内
        'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3],...
        'Layer', 'top')  % 确保坐标轴在顶层

% 轴标签设置
xlabel('Intersection Number', 'FontSize', 14, 'FontWeight', 'normal')
ylabel('RMOTA Improvement Percentage', 'FontSize', 14, 'FontWeight', 'normal')
ylim([0 100])

% 专业图例设置
legend({'Town01', 'Town10'}, 'Location', 'northeast', 'FontSize', 13,...
       'Box', 'off', 'EdgeColor', 'none')

% 标题设置
title('Multi-Scenario RMOTA Performance Comparison',...
      'FontSize', 16, 'FontWeight', 'normal', 'Color', [0.15 0.15 0.15])

% 导出高清图像
%exportgraphics(gcf, 'Professional_RMOTA_Chart.png', 'Resolution', 600)
exportgraphics(gcf, 'Professional_RMOTA_Chart.eps', 'ContentType','vector');